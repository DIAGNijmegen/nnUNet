from collections import OrderedDict

from batchgenerators.transforms.spatial_transforms import MirrorTransform
from typing import Dict, Type
from pathlib import Path

import pytest
import torch
import numpy as np

from batchgenerators.transforms.channel_selection_transforms import (
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.utility_transforms import (
    RenameTransform,
    RemoveLabelTransform,
    NumpyToTensor,
)

import nnunet
from nnunet.training.data_augmentation.diag.data_augmentation_moreDA_sparse import get_moreDA_augmentation_sparse
from nnunet.training.data_augmentation.diag.transforms.spatial_transforms import SparseSpatialTransform
from nnunet.training.data_augmentation.downsampling import (
    DownsampleSegForDSTransform2,
    downsample_seg_for_ds_transform2,
)
from nnunet.training.dataloading.dataset_loading import (
    load_dataset, DataLoader3D,
)
from nnunet.training.network_training.diag.nnUNetTrainerV2Sparse import nnUNetTrainerV2Sparse, \
    nnUNetTrainerV2SparseNormalSampling
import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.run.default_configuration as nndc
from nnunet.paths import default_plans_identifier

from tests.test_training import check_expected_training_output, prepare_paths


RESOURCES_DIR = Path(__file__).parent / "resources"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
HIPPOCAMPUS_TASK_ID = 4
HIPPOCAMPUS_TASK = "Task004_Hippocampus"


def test_sparse_data_augmentation():

    # NOTE: strange that random_crop augmentation is set to False by default,
    # this seems like a sensible default augmentation to enable.
    # anyway random_crop is used (within mask) by the sparse trainer anyway by default, so shouldn't be a problem...

    data_aug_params = {
        "selected_data_channels": None,
        "selected_seg_channels": [0],
        "do_elastic": False,
        "elastic_deform_alpha": (0.0, 900.0),
        "elastic_deform_sigma": (9.0, 13.0),
        "p_eldef": 0.2,
        "do_scaling": True,
        "scale_range": (0.7, 1.4),
        "independent_scale_factor_for_each_axis": False,
        "p_independent_scale_per_axis": 1,
        "p_scale": 0.2,
        "do_rotation": True,
        "rotation_x": (-0.5235987755982988, 0.5235987755982988),
        "rotation_y": (-0.5235987755982988, 0.5235987755982988),
        "rotation_z": (-0.5235987755982988, 0.5235987755982988),
        "rotation_p_per_axis": 1,
        "p_rot": 0.2,
        "random_crop": False,
        "random_crop_dist_to_border": None,
        "do_gamma": True,
        "gamma_retain_stats": True,
        "gamma_range": (0.7, 1.5),
        "p_gamma": 0.3,
        "do_mirror": True,
        "mirror_axes": (0, 1, 2),
        "dummy_2D": False,
        "mask_was_used_for_normalization": OrderedDict([(0, False)]),
        "border_mode_data": "constant",
        "all_segmentation_labels": None,
        "move_last_seg_chanel_to_data": False,
        "cascade_do_cascade_augmentations": False,
        "cascade_random_binary_transform_p": 0.4,
        "cascade_random_binary_transform_p_per_label": 1,
        "cascade_random_binary_transform_size": (1, 8),
        "cascade_remove_conn_comp_p": 0.2,
        "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
        "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,
        "do_additive_brightness": False,
        "additive_brightness_p_per_sample": 0.15,
        "additive_brightness_p_per_channel": 0.5,
        "additive_brightness_mu": 0.0,
        "additive_brightness_sigma": 0.1,
        "num_threads": 12,
        "num_cached_per_thread": 2,
        "patch_size_for_spatialtransform": np.array([40, 56, 40]),
    }
    p = (
        NNUNET_PREPROCESSING_DATA_DIR
        / HIPPOCAMPUS_TASK
        / "nnUNetData_plans_v2.1_stage0"
    )
    dataset = load_dataset(str(p))
    dl_tr = DataLoader3D(
        dataset,
        (73, 80, 64),
        (40, 56, 40),
        9,
        False,
        oversample_foreground_percent=0.33,
        pad_mode="constant",
        pad_sides=None,
        memmap_mode="r",
    )
    dl_val = DataLoader3D(
        dataset,
        (40, 56, 40),
        (40, 56, 40),
        9,
        False,
        oversample_foreground_percent=0.33,
        pad_mode="constant",
        pad_sides=None,
        memmap_mode="r",
    )
    (
        tr_gen,
        val_gen,
    ) = get_moreDA_augmentation_sparse(
        dl_tr,
        dl_val,
        data_aug_params["patch_size_for_spatialtransform"],
        data_aug_params,
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        pin_memory=True,
        use_nondetMultiThreadedAugmenter=False,
    )
    all_transforms = tr_gen.transform.transforms
    tr_gen.transform.transforms = []
    data_dict = next(tr_gen)
    initial_data = np.copy(data_dict["data"])
    initial_seg = np.copy(data_dict["seg"])

    def assert_data_properties(
        data_dict: Dict,
        data_key: str = "data",
        seg_key: str = "seg",
        seg_min: float = -1.0,
    ):
        assert data_dict[data_key].min() < -1.0
        assert data_dict[data_key].max() > 6.0
        assert data_dict[seg_key].min() == seg_min
        assert data_dict[seg_key].max() == 2.0
        assert all(
            [
                data_dict[key].dtype == np.float32
                for key in [data_key, seg_key]
            ]
        )

    assert_data_properties(data_dict=data_dict)

    def print_stats(e):
        print(e.shape, np.min(e), np.max(e))

    for idx, tf in enumerate(all_transforms):
        print(f"{idx:2} Applying transform: {tf}")
        if "data" in data_dict:
            print_stats(data_dict["data"])
        if "seg" in data_dict:
            print_stats(data_dict["seg"])
        if "target" in data_dict:
            if isinstance(data_dict["target"], list):
                for e in data_dict["target"]:
                    print_stats(e)
            else:
                print_stats(data_dict["target"])

        data_dict = tf(**data_dict)
        assert not isinstance(tf, RemoveLabelTransform)
        if idx == 0:
            assert isinstance(tf, SegChannelSelectionTransform)
            assert np.array_equal(initial_data, data_dict["data"])
            assert np.array_equal(initial_seg, data_dict["seg"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 1:
            assert isinstance(tf, SparseSpatialTransform)
            assert not np.array_equal(initial_data, data_dict["data"])
            assert not np.array_equal(initial_seg, data_dict["seg"])
            data_ref = np.copy(data_dict["data"])
            seg_ref = np.copy(data_dict["seg"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 8:
            assert isinstance(tf, GammaTransform)
            assert not np.array_equal(data_ref, data_dict["data"])
            assert np.array_equal(seg_ref, data_dict["seg"])
            data_ref = np.copy(data_dict["data"])
            seg_ref = np.copy(data_dict["seg"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 9:
            assert isinstance(tf, MirrorTransform)
            matrices = [data_ref, seg_ref]
            keys = ["data", "seg"]
            batch_size = data_dict["data"].shape[0]
            found_valid_flip = [False for _ in range(batch_size)]
            # per batch element 8 options for flipping based on dimensions...
            for b in range(batch_size):
                for flipx in ((), (1,)):
                    for flipy in ((), (2,)):
                        for flipz in ((), (3,)):
                            flips = flipx + flipy + flipz
                            if all(
                                [
                                    np.all(
                                        np.flip(m[b, :], flips) == data_dict[key][b, :]
                                    )
                                    for m, key in zip(matrices, keys)
                                ]
                            ):
                                found_valid_flip[b] = True
            assert all(found_valid_flip)
            assert_data_properties(data_dict=data_dict)
        elif idx == 11:
            assert isinstance(tf, RenameTransform)
            assert all([key in data_dict for key in ["data", "target"]])
            assert "seg" not in data_dict
            assert_data_properties(data_dict=data_dict, seg_key="target")
            seg_ref = np.copy(data_dict["target"])
        elif idx == 12:
            assert isinstance(tf, DownsampleSegForDSTransform2)
            for order, (img, key) in enumerate(
                zip([seg_ref], ["target"])
            ):
                assert isinstance(data_dict[key], list)
                assert len(data_dict[key]) == 3
                for idx, scales in enumerate(
                    ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                ):
                    assert np.array_equal(
                        data_dict[key][idx],
                        downsample_seg_for_ds_transform2(
                            seg=img, ds_scales=(scales,), order=order, axes=None,
                        )[0],
                    )
        elif idx == 13:
            assert isinstance(tf, NumpyToTensor)
            assert isinstance(data_dict["data"], torch.Tensor)
            assert data_dict["data"].min() < -1.0
            assert data_dict["data"].max() > 6.0
            for i in range(3):
                assert isinstance(data_dict["target"][i], torch.Tensor)
                assert data_dict["target"][i].min() == -1.0
                assert data_dict["target"][i].max() == 2.0
        elif idx > 14:
            raise RuntimeError("No more than 15 transforms were expected...")


@pytest.mark.parametrize("trainer_class", (nnUNetTrainerV2Sparse, nnUNetTrainerV2SparseNormalSampling))
@pytest.mark.parametrize("network", ("3d_fullres",))
@pytest.mark.parametrize("fold", (0,))
def test_sparse_trainer(tmp_path: Path, network: str, fold: int, trainer_class: Type[nnUNetTrainerV2Sparse]):
    prepare_paths(output_dir=tmp_path)
    task = nnp.convert_id_to_task_name(HIPPOCAMPUS_TASK_ID)
    decompress_data = True
    deterministic = False
    run_mixed_precision = True
    (
        plans_file,
        output_folder_name,
        dataset_directory,
        batch_dice,
        stage,
        trainer_class,
    ) = nndc.get_default_configuration(
        network=network,
        task=task,
        network_trainer=trainer_class.__name__,
        plans_identifier=default_plans_identifier,
    )
    trainer = trainer_class(
        plans_file=plans_file,
        fold=fold,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )
    normal_sampling_trainer = isinstance(trainer, nnUNetTrainerV2SparseNormalSampling)
    assert isinstance(trainer, trainer_class)
    assert trainer.only_sample_from_annotated != normal_sampling_trainer

    trainer.max_num_epochs = 2
    trainer.num_batches_per_epoch = 2
    trainer.num_val_batches_per_epoch = 2
    trainer.initialize(True)
    assert any([isinstance(tf, SparseSpatialTransform) for tf in trainer.tr_gen.transform.transforms]) != normal_sampling_trainer

    trainer.run_training()
    trainer.network.eval()
    trainer.validate(
        save_softmax=False,
        validation_folder_name="validation_raw",
        run_postprocessing_on_folds=True,
        overwrite=True,
    )
    check_expected_training_output(
        check_dir=tmp_path,
        network=network,
        trainer_class_name=trainer_class.__name__,
    )
