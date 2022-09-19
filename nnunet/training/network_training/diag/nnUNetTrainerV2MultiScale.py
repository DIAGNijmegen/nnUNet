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
from multiprocessing import Pool
from time import sleep
from typing import Tuple

import numpy as np
import torch
import wandb
from batchgenerators.utilities.file_and_folder_operations import *
from mtdp import build_model
from scipy.stats import zscore
from torch import nn
from torch.cuda.amp import autocast

from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.nnUNet_variants.generic_UNet_multiScale import GenericUNetMultiScale
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.data_augmentation_moreDA_pathology_no_spatial import \
    get_moreDA_augmentation_pathology_no_spatial
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.dataloading.diag.dataset_loading_insideroi_multiscale import DataLoader2DROIsMultiScale
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities import shutil_sol
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class nnUNetTrainerV2MultiScale(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, data_origin=None, labels_dict=None, spacing=8.0,
                 target_spacing=0.5, encoder_kwargs=None, convolutional_pooling=True):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.convolutional_pooling = convolutional_pooling
        self.spacing = spacing
        self.target_spacing = target_spacing
        self.labels_dict = labels_dict
        self.data_origin = data_origin
        self.save_every = 5
        if encoder_kwargs is None:
            print("Since encoder_kwargs is empty, we will use the default: resnet18 with imagenet weights.")
            encoder_kwargs = {"arch": "resnet18", "pretrained": "imagenet"}
        self.encoder_kwargs = encoder_kwargs
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.dl_val_full = None

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            encoder_trainable = self.encoder_kwargs.pop("trainable", True)
            self.encoder = build_model(**self.encoder_kwargs)
            if not encoder_trainable:
                self.encoder.eval()
            self.process_plans(self.plans)
            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.prepare_data()
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def prepare_data(self):
        self.dl_tr, self.dl_val, self.dl_val_full = self.get_basic_generators()
        if self.unpack_data:
            print("unpacking dataset")
            unpack_dataset(self.folder_with_preprocessed_data)
            print("done")
        else:
            print(
                "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                "will wait all winter for your model to finish!")
        # Only color and simple spatial transforms. Otherwise misalignment of the context and target patches could happen
        self.tr_gen, self.val_gen = get_moreDA_augmentation_pathology_no_spatial(
            self.dl_tr, self.dl_val,
            params=self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory
        )
        _, self.val_gen_full_size = get_moreDA_augmentation_pathology_no_spatial(
            self.dl_tr, self.dl_val_full,
            params=self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory
        )
        self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                               also_print_to_console=False)
        self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                               also_print_to_console=False)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_tr = DataLoader2DROIsMultiScale(
            data_origin=self.data_origin,
            spacing=self.spacing,
            data=self.dataset_tr,
            final_patch_size=self.basic_generator_patch_size,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode="constant",
            pad_sides=self.pad_all_sides,
            memmap_mode='r'
        )
        dl_val = DataLoader2DROIsMultiScale(
            data_origin=self.data_origin,
            spacing=self.spacing,
            data=self.dataset_val,
            final_patch_size=self.patch_size,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode="constant",
            pad_sides=self.pad_all_sides,
            memmap_mode='r',
            training=True,
            crop_to_patch_size=True
        )
        dl_val_full = DataLoader2DROIsMultiScale(
            data_origin=self.data_origin,
            spacing=self.spacing,
            data=self.dataset_val,
            final_patch_size=self.patch_size,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode="constant",
            pad_sides=self.pad_all_sides,
            memmap_mode='r',
            training=False,
            crop_to_patch_size=False
        )

        return dl_tr, dl_val, dl_val_full

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = GenericUNetMultiScale(self.encoder, self.spacing, self.target_spacing, self.num_input_channels,
                                             self.base_num_features,
                                             self.num_classes, len(self.net_num_pool_op_kernel_sizes),
                                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                             dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, True, False, lambda x: x,
                                             InitWeights_He(1e-2), self.net_num_pool_op_kernel_sizes,
                                             self.net_conv_kernel_sizes, False, self.convolutional_pooling,
                                             self.convolutional_pooling)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        if not self.val_gen:
            self.prepare_data()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        for data_dict in self.val_gen_full_size:
            properties = data_dict['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = data_dict['data'][0]

                print(data_dict['keys'], data.shape)

                pred, softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data,
                                                                                           do_mirroring=do_mirroring,
                                                                                           mirror_axes=mirror_axes,
                                                                                           use_sliding_window=use_sliding_window,
                                                                                           step_size=step_size,
                                                                                           use_gaussian=use_gaussian,
                                                                                           all_in_gpu=all_in_gpu,
                                                                                           mixed_precision=self.fp16)

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating objects
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil_sol.copyfile(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        ds = self.network.do_ds
        self.network.do_ds = False
        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.do_ds = ds
        self.network.train(current_mode)
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.data_aug_params["do_scaling"] = False
        self.data_aug_params["do_rotation"] = False
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params["do_gamma"] = False
        self.data_aug_params["do_additive_brightness"] = True
        self.data_aug_params["do_mirror"] = True
        self.basic_generator_patch_size = self.patch_size
        self.data_aug_params["do_hed"] = True
        self.data_aug_params["hed_params"] = dict(factor=0.05)
        self.data_aug_params["do_hsv"] = False
        self.data_aug_params["hsv_params"] = dict(h_lim=0.01, s_lim=0.01, v_lim=0.05)
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        config = self.get_debug_information()
        wandb.init(project=os.environ.get("WANDB_PROJECT"), config=config)

        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret

    @staticmethod
    def z_score_norm(array: np.ndarray):
        return zscore(array.reshape(-1, 3)).reshape(array.shape)

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network,
                                   torch.rand((1, 2, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network,
                                   torch.rand((1, 2, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        self.log_to_wandb()
        return continue_training

    def log_to_wandb(self):
        wandb.log({
            "train/loss": self.all_tr_losses[-1],
            "val/loss": self.all_val_losses[-1],
            **{f'val/metric_{cls_index}': score for cls_index, score in
               self.all_val_eval_metrics_per_class[-1].items() if not np.isnan(score)},
            "val/metric": self.all_val_eval_metrics[-1],
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })
