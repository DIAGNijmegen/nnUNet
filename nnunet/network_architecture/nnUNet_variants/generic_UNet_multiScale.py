#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional
from batchgenerators.augmentations.utils import pad_nd_image
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch import nn
from torch.cuda.amp import autocast
from torchvision.transforms.functional import center_crop

from nnunet.network_architecture.generic_UNet import Generic_UNet, StackedConvLayers, ConvDropoutNormNonlin, Upsample
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class GenericUNetMultiScale(Generic_UNet):
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648

    def __init__(self, context_encoder: torch.nn.Module, ct_spacing, tg_spacing, context_num_classes, use_context,
                 *args, **kwargs):
        """
        First version that uses a CNN to encode context around the main patch. Assuming that the following formula holds:
        Encoder contains 5 downsamplings and the input patch is at 8.0 mpp. This results in 2**3 * 2**5 = 2**8 mpp patch
        of 16x16xFM (this depends on resnet18/50). nnUNet will be fixed at 7 poolings to allow for the encoder in VRAM.
        With 7 poolings the input of 0.5 mpp will be 2**-1 * 2**7 = 2**6 mpp. To braze the gap the encoded patch will be
        cropped by 2**2 = 2**(8-6).

        NOTE: most of the code is by nnU-Net with only slide adaptations to allow for a context patch.
        """
        super(GenericUNetMultiScale, self).__init__(*args, **kwargs)
        self.context_num_classes = context_num_classes
        self.context_encoder = context_encoder
        self.use_context = use_context
        reduce_conv_kwargs = self.conv_kwargs.copy()
        reduce_conv_kwargs['kernel_size'] = [1, 1]
        reduce_conv_kwargs['padding'] = [0, 0]

        input_dim = 512
        if self.convolutional_pooling:
            b_t, c_t, self.w_t, self.h_t = nn.Sequential(*self.conv_blocks_context)(
                torch.zeros((1, 3, input_dim, input_dim))).shape
        else:
            b_t, c_t, _, _ = nn.Sequential(*self.conv_blocks_context)(torch.zeros((1, 3, input_dim, input_dim))).shape
            _, _, self.w_t, self.h_t = nn.Sequential(*self.td)(torch.zeros((1, 3, input_dim, input_dim))).shape
        b_e, c_e, self.w_e, self.h_e = self.context_encoder(torch.zeros((1, 3, input_dim, input_dim))).shape
        context_ds = np.log2(input_dim) - np.log2(self.w_e)
        target_ds = np.log2(input_dim) - np.log2(self.w_t)
        spatial_difference = np.log2(self.w_e) - np.log2(self.w_t)
        resolutional_difference = (np.log2(ct_spacing) + context_ds) - (np.log2(tg_spacing) + target_ds)
        # Computes the required amount of down samplings w.r.t. the context encoder
        # in order to align target and context patch, before  interpolating to the target patch size
        self.ds_difference = int(resolutional_difference) + int(spatial_difference)
        self.upsample = Upsample(scale_factor=2 ** self.ds_difference, mode="bicubic")
        self.reduce_fm_conv = StackedConvLayers(c_t + c_e, c_t, 1,
                                                self.conv_op, reduce_conv_kwargs, self.norm_op,
                                                self.norm_op_kwargs, self.dropout_op,
                                                self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                basic_block=ConvDropoutNormNonlin)
        reduce_spatial_conv_kwargs = self.conv_kwargs.copy()
        reduce_spatial_conv_kwargs['kernel_size'] = (self.w_e - 1, self.h_e - 1)
        reduce_spatial_conv_kwargs['padding'] = [0, 0]
        if self.context_num_classes:
            self.context_logits_conv = nn.AdaptiveAvgPool2d((1, 1))
            self.context_logits_fc = nn.Sequential(
                *[nn.Linear(c_e, self.context_num_classes), nn.Dropout(p=0.25)]
            )
        self.counter = 0
        del b_t, c_t, b_e, c_e, self.w_t, self.h_t, self.w_e, self.h_e

    def forward(self, x):
        """
        :param x: shape should be (batch number, target + context, channels, width, height)
        """
        main = x[:, 0]
        context = x[:, 1]

        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            main = self.conv_blocks_context[d](main)
            skips.append(main)
            if not self.convolutional_pooling:
                main = self.td[d](main)

        main_encoding = self.conv_blocks_context[-1](main)
        if self.use_context:
            context_encoding = self.context_encoder(context)

            w, h = context_encoding.shape[-2:]

            cropped = center_crop(context_encoding, output_size=[
                int(w * 2 ** -self.ds_difference),
                int(h * 2 ** -self.ds_difference)
            ])

            upsampled = nn.functional.interpolate(cropped, size=list(main_encoding.shape[-2:]), mode='bicubic')
            x = torch.cat((upsampled, main_encoding), dim=1)

            x = self.reduce_fm_conv(x)

            context_logits = None
            if self.context_num_classes:
                context_encoding = self.context_logits_conv(context_encoding)
                context_logits = self.context_logits_fc(context_encoding.flatten(start_dim=1))
        else:
            x = main_encoding
            context_logits = None
        del context, main_encoding

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1],
                                                  seg_outputs[:-1][::-1])]), context_logits
        else:
            return seg_outputs[-1], context_logits

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: Union[np.ndarray, torch.tensor] = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, modality, c, x, y)'

        color_values = ['yellow', 'firebrick', 'pink', 'purple', "red", "green", "blue", "white", "white"]
        cmap = ListedColormap(color_values)
        bounds = list(range(len(color_values)))
        norm = BoundaryNorm(bounds, cmap.N, clip=True)

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[3:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        # fig, axs = plt.subplots(5, 3, figsize=(15, 25))
        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x)[0])
                # axs[0][0].imshow(x[0][0].detach().cpu().numpy().transpose(1, 2, 0))
                # axs[0][1].imshow(pred.detach().cpu().numpy()[0].argmax(axis=0), cmap=cmap, norm=norm,
                #                  interpolation='nearest')
                # axs[0][2].imshow(x[0][1].detach().cpu().numpy().transpose(1, 2, 0))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4,)))[0])
                # axs[1][0].imshow(torch.flip(x, (4,))[0][0].detach().cpu().numpy().transpose(1, 2, 0))
                # axs[1][1].imshow(pred.detach().cpu().numpy()[0].argmax(axis=0), cmap=cmap, norm=norm,
                #                  interpolation='nearest')
                # axs[1][2].imshow(torch.flip(x, (4,))[0][1].detach().cpu().numpy().transpose(1, 2, 0))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3,)))[0])
                # axs[2][0].imshow(torch.flip(x, (3,))[0][0].detach().cpu().numpy().transpose(1, 2, 0))
                # axs[2][1].imshow(pred.detach().cpu().numpy()[0].argmax(axis=0), cmap=cmap, norm=norm,
                #                  interpolation='nearest')
                # axs[2][2].imshow(torch.flip(x, (3,))[0][1].detach().cpu().numpy().transpose(1, 2, 0))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)))[0])
                # axs[3][0].imshow(torch.flip(x, (4, 3))[0][0].detach().cpu().numpy().transpose(1, 2, 0))
                # axs[3][1].imshow(pred.detach().cpu().numpy()[0].argmax(axis=0), cmap=cmap, norm=norm,
                #                  interpolation='nearest')
                # axs[3][2].imshow(torch.flip(x, (4, 3))[0][1].detach().cpu().numpy().transpose(1, 2, 0))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if mult is not None:
                result_torch[:, :] *= mult
        #
        # axs[4][0].imshow(x[0][0].detach().cpu().numpy().transpose(1, 2, 0))
        # axs[4][1].imshow(result_torch.detach().cpu().numpy()[0].argmax(axis=0), cmap=cmap, norm=norm,
        #                  interpolation='nearest')
        # axs[4][2].imshow(torch.flip(x, (4, 3))[0][1].detach().cpu().numpy().transpose(1, 2, 0))
        # plt.tight_layout()
        # plt.savefig(os.path.join(os.environ['RESULTS_FOLDER'], f"{self.counter}.png"))
        # plt.close()
        # self.counter += 1
        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (modality, c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        context = x[1]
        main = x[0]

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(main, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)
            context = torch.from_numpy(context).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        data = maybe_to_torch(data)
        context = maybe_to_torch(context)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    torch.stack([data[:, lb_x:ub_x, lb_y:ub_y], context[:, lb_x:ub_x, lb_y:ub_y]])[None],
                    mirror_axes,
                    do_mirroring,
                    gaussian_importance_map
                )[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv_tiled(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                          mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                          regions_class_order: tuple = None, use_gaussian: bool = False,
                                          pad_border_mode: str = "edge", pad_kwargs: dict = None,
                                          all_in_gpu: bool = False,
                                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        predicted_segmentation = []
        softmax_pred = []
        x = x[:, None]

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        x = x[:, None]
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(
                x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])

        predicted_probabilities = predicted_probabilities[:, slicer[1], slicer[2]]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None,
                   regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (modality,c,x,y)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes,
                                                                 step_size,
                                                                 regions_class_order, use_gaussian, pad_border_mode,
                                                                 pad_kwargs, all_in_gpu, False)
                else:
                    res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes,
                                                           regions_class_order,
                                                           pad_border_mode, pad_kwargs, all_in_gpu, False)
        return res
