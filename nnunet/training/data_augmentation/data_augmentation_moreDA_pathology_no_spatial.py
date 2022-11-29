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
import albumentations as A
import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.diag.transforms.custom_transforms import MirrorTransform2D
from nnunet.training.data_augmentation.diag.transforms.pathology_color_transforms import HedTransform, HsvTransform
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


class AlbuRandomRotate90:

    def __init__(self, data_key="data", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key
        self.rotate = A.ReplayCompose([A.RandomRotate90(p=1.0)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                s = seg[b].transpose(1, 2, 0)
                init = self.rotate(image=d[0], mask=s)
                new_d = np.zeros_like(d)
                new_d[0] = init['image']
                s = init['mask'].transpose(2, 0, 1)
                for i in range(1, d.shape[0]):
                    new_d[i] = A.ReplayCompose.replay(init['replay'], image=d[i])['image']
                d = new_d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                s = seg[b].transpose(1, 2, 0)
                augmented = self.rotate(image=d, mask=s)
                d = augmented['image'].transpose(2, 0, 1)
                s = augmented['mask'].transpose(2, 0, 1)
            data[b] = d
            seg[b] = s
        data_dict[self.data_key] = data
        data_dict[self.seg_key] = seg
        return data_dict


class AlbuTranspose:

    def __init__(self, data_key="data", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key
        self.aug = A.ReplayCompose([A.Transpose(p=.5)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                s = seg[b].transpose(1, 2, 0)
                init = self.aug(image=d[0], mask=s)
                new_d = np.zeros_like(d)
                new_d[0] = init['image']
                s = init['mask'].transpose(2, 0, 1)
                for i in range(1, d.shape[0]):
                    new_d[i] = A.ReplayCompose.replay(init['replay'], image=d[i])['image']
                d = new_d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                s = seg[b].transpose(1, 2, 0)
                augmented = self.aug(image=d, mask=s)
                d = augmented['image'].transpose(2, 0, 1)
                s = augmented['mask'].transpose(2, 0, 1)
            data[b] = d
            seg[b] = s
        data_dict[self.data_key] = data
        data_dict[self.seg_key] = seg
        return data_dict


class AlbuMirror:

    def __init__(self, data_key="data", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key
        self.aug = A.ReplayCompose([A.Flip(p=.5)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                s = seg[b].transpose(1, 2, 0)
                init = self.aug(d=np.random.choice([-1, 0, 1]), image=d[0], mask=s)
                new_d = np.zeros_like(d)
                new_d[0] = init['image']
                s = init['mask'].transpose(2, 0, 1)
                for i in range(1, d.shape[0]):
                    new_d[i] = A.ReplayCompose.replay(init['replay'], image=d[i])['image']
                d = new_d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                s = seg[b].transpose(1, 2, 0)
                augmented = self.aug(image=d, mask=s)
                d = augmented['image'].transpose(2, 0, 1)
                s = augmented['mask'].transpose(2, 0, 1)
            data[b] = d
            seg[b] = s
        data_dict[self.data_key] = data
        data_dict[self.seg_key] = seg
        return data_dict


class AlbuGaussianBlur:

    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.aug = A.ReplayCompose([A.GaussianBlur((1, 3), p=1.0)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                init = self.aug(image=d[0])
                d = np.stack([init['image']] + [A.ReplayCompose.replay(init['replay'], image=d[i])['image'] for i in
                                                range(1, d.shape[0])])
                d = d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                d = self.aug(image=d)['image']
                d = d.transpose(2, 0, 1)
            data[b] = d
        data_dict[self.data_key] = data
        return data_dict


class AlbuGaussionNoise:

    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.aug = A.ReplayCompose([A.GaussNoise(p=1.0)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                init = self.aug(image=d[0])
                d = np.stack([init['image']] + [A.ReplayCompose.replay(init['replay'], image=d[i])['image'] for i in
                                                range(1, d.shape[0])])
                d = d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                d = self.aug(image=d)['image']
                d = d.transpose(2, 0, 1)
            data[b] = d
        data_dict[self.data_key] = data
        return data_dict


class AlbuRandomBrightness:

    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.aug = A.ReplayCompose([A.ColorJitter(brightness=0.1, contrast=0.05, saturation=0, p=1.0)])

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)

        for b in range(data.shape[0]):
            if data[b].ndim == 4:
                d = data[b].transpose(0, 2, 3, 1)
                init = self.aug(image=d[0])
                d = np.stack([init['image']] + [A.ReplayCompose.replay(init['replay'], image=d[i])['image'] for i in
                                                range(1, d.shape[0])])
                d = d.transpose(0, 3, 1, 2)
            else:
                d = data[b].transpose(1, 2, 0)
                d = self.aug(image=d)['image']
                d = d.transpose(2, 0, 1)
            data[b] = d
        data_dict[self.data_key] = data
        return data_dict


def get_moreDA_augmentation_pathology_no_spatial(dataloader_train, dataloader_val,
                                                 params=default_3D_augmentation_params,
                                                 seeds_train=None, seeds_val=None, deep_supervision_scales=None,
                                                 soft_ds=False,
                                                 classes=None, pin_memory=True, regions=None,
                                                 use_nondetMultiThreadedAugmenter: bool = False):
    """
    Removed all transformations that cause misalignment of the target and context patch.
    """

    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    tr_transforms.append(AlbuGaussianBlur())
    tr_transforms.append(AlbuRandomBrightness())
    tr_transforms.append(AlbuRandomRotate90())
    tr_transforms.append(AlbuTranspose())

    if params.get("do_hed"):
        tr_transforms.append(HedTransform(**params["hed_params"]))
    if params.get("do_hsv"):
        tr_transforms.append(HsvTransform(**params["hsv_params"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform2D(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                    p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get("num_cached_per_thread"), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val
