import pickle

from collections import OrderedDict
from typing import Dict
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
from nnunet.training.data_augmentation.diag.data_augmentation_moreDA_weightmaps import \
    get_moreDA_augmentation_weightmaps
from nnunet.training.data_augmentation.downsampling import (
    DownsampleSegForDSTransform2,
    downsample_seg_for_ds_transform2,
)
from nnunet.training.dataloading.dataset_loading import (
    load_dataset,
    unpack_dataset,
    DataLoader3D,
    DataLoader2D,
)
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.dataloading.diag.dataset_loading_weightmaps import DataLoader3DWeighted
from nnunet.training.network_training.diag.nnUNetTrainerV2Weighted import (
    nnUNetTrainerV2Weighted,
    DC_and_CE_loss_weighted,
)
from nnunet.training.data_augmentation.diag.transforms.spatial_transforms_weightmaps import (
    SpatialTransformWithWeights,
    MirrorTransformWithWeights,
)
import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.run.default_configuration as nndc
from nnunet.paths import default_plans_identifier
from nnunet.utilities.diag.generate_unet_weightmaps import (
    generate_unet_weightmaps,
    get_classes_from_dataset_file,
)

from tests.test_training import check_expected_training_output, prepare_paths
from tests.utils import is_data_integrity_ok_md5sum

RESOURCES_DIR = Path(__file__).parent / "resources"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
HIPPOCAMPUS_TASK_ID = 4
HIPPOCAMPUS_TASK = "Task004_Hippocampus"


def test_bootstrapping_weightmaps(tmp_path):
    input_dir = (
        NNUNET_PREPROCESSING_DATA_DIR
        / HIPPOCAMPUS_TASK
        / "nnUNetData_plans_v2.1_stage0"
    )
    output_dir = Path(tmp_path / HIPPOCAMPUS_TASK / "nnUNetData_plans_v2.1_stage0")
    output_dir.mkdir(parents=True)
    classes = get_classes_from_dataset_file(
        input_dir=input_dir, background_label=0
    )
    generate_unet_weightmaps(
        input_dir=input_dir,
        output_dir=output_dir,
        matching_pattern="*.npz",
        sigma=5.0,
        w_0=10.0,
        classes=classes,
    )
    assert is_data_integrity_ok_md5sum(
        workdir=tmp_path,
        md5file=NNUNET_PREPROCESSING_DATA_DIR / "Task004_Hippocampus_weightmaps.md5",
    )


def test_dataset_loading():
    p = (
        NNUNET_PREPROCESSING_DATA_DIR
        / HIPPOCAMPUS_TASK
        / "nnUNetData_plans_v2.1_stage0"
    )
    dataset = load_dataset(str(p))
    with open(
        NNUNET_PREPROCESSING_DATA_DIR
        / HIPPOCAMPUS_TASK
        / "nnUNetPlansv2.1_plans_3D.pkl",
        "rb",
    ) as f:
        plans = pickle.load(f)
    patch_size = plans["plans_per_stage"][0]["patch_size"]
    unpack_dataset(str(p))
    d2 = DataLoader3D(
        dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33
    )
    normal_data = next(d2)
    assert "weightmap" not in normal_data
    d1 = DataLoader3DWeighted(
        dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33
    )
    weighted_data = next(d1)
    assert "weightmap" in weighted_data
    assert np.sum(weighted_data["weightmap"] > 0)

    DataLoader3D(
        dataset,
        np.array(patch_size).astype(int),
        np.array(patch_size).astype(int),
        2,
        oversample_foreground_percent=0.33,
    )
    DataLoader2D(
        dataset,
        (64, 64),
        np.array(patch_size).astype(int)[1:],
        12,
        oversample_foreground_percent=0.33,
    )


def test_weighted_data_augmentation():
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
    dl_tr = DataLoader3DWeighted(
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
    dl_val = DataLoader3DWeighted(
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
    ) = get_moreDA_augmentation_weightmaps(
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
    initial_weightmaps = np.copy(data_dict["weightmap"])

    def assert_data_properties(
        data_dict: Dict,
        data_key: str = "data",
        seg_key: str = "seg",
        weightmap_key: str = "weightmap",
        seg_min: float = -1.0,
    ):
        assert data_dict[data_key].min() < -1.0
        assert data_dict[data_key].max() > 6.0
        assert data_dict[seg_key].min() == seg_min
        assert data_dict[seg_key].max() == 2.0
        assert data_dict[weightmap_key].min() == 0.0
        assert data_dict[weightmap_key].max() > 5.0
        assert all(
            [
                data_dict[key].dtype == np.float32
                for key in [data_key, seg_key, weightmap_key]
            ]
        )

    assert_data_properties(data_dict=data_dict)

    for idx, tf in enumerate(all_transforms):
        print(f"{idx:2} Applying transform: {tf}")
        data_dict = tf(**data_dict)
        if idx == 0:
            assert isinstance(tf, SegChannelSelectionTransform)
            assert np.array_equal(initial_data, data_dict["data"])
            assert np.array_equal(initial_seg, data_dict["seg"])
            assert np.array_equal(initial_weightmaps, data_dict["weightmap"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 1:
            assert isinstance(tf, SpatialTransformWithWeights)
            assert not np.array_equal(initial_data, data_dict["data"])
            assert not np.array_equal(initial_seg, data_dict["seg"])
            assert not np.array_equal(initial_weightmaps, data_dict["weightmap"])
            data_ref = np.copy(data_dict["data"])
            seg_ref = np.copy(data_dict["seg"])
            weightmap_ref = np.copy(data_dict["weightmap"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 8:
            assert isinstance(tf, GammaTransform)
            assert not np.array_equal(data_ref, data_dict["data"])
            assert np.array_equal(seg_ref, data_dict["seg"])
            assert np.array_equal(weightmap_ref, data_dict["weightmap"])
            data_ref = np.copy(data_dict["data"])
            seg_ref = np.copy(data_dict["seg"])
            weightmap_ref = np.copy(data_dict["weightmap"])
            assert_data_properties(data_dict=data_dict)
        elif idx == 9:
            assert isinstance(tf, MirrorTransformWithWeights)
            matrices = [data_ref, seg_ref, weightmap_ref]
            keys = ["data", "seg", "weightmap"]
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
            assert isinstance(tf, RemoveLabelTransform)
            assert_data_properties(data_dict=data_dict, seg_min=0.0)
        elif idx == 12:
            assert isinstance(tf, RenameTransform)
            assert all([key in data_dict for key in ["data", "target", "weightmap"]])
            assert "seg" not in data_dict
            assert_data_properties(data_dict=data_dict, seg_key="target", seg_min=0.0)
            seg_ref = np.copy(data_dict["target"])
            weightmap_ref = np.copy(data_dict["weightmap"])
        elif idx == 14:
            assert isinstance(tf, DownsampleSegForDSTransform2)
            for order, (img, key) in enumerate(
                zip([seg_ref, weightmap_ref], ["target", "weightmap"])
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
        elif idx == 15:
            assert isinstance(tf, NumpyToTensor)
            assert isinstance(data_dict["data"], torch.Tensor)
            assert data_dict["data"].min() < -1.0
            assert data_dict["data"].max() > 6.0
            for i in range(3):
                assert isinstance(data_dict["target"][i], torch.Tensor)
                assert data_dict["target"][i].min() == 0.0
                assert data_dict["target"][i].max() == 2.0
                assert isinstance(data_dict["weightmap"][i], torch.Tensor)
                assert data_dict["weightmap"][i].min() == 0.0
                assert data_dict["weightmap"][i].max() > 5.0
        elif idx > 15:
            raise RuntimeError("No more than 16 transforms were expected...")


@pytest.mark.parametrize("network", ("3d_fullres",))
@pytest.mark.parametrize("fold", (0,))
def test_weighted_trainer(tmp_path: Path, network: str, fold: int):
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
        network_trainer=nnUNetTrainerV2Weighted.__name__,
        plans_identifier=default_plans_identifier,
    )
    assert issubclass(trainer_class, nnUNetTrainerV2Weighted)
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
    trainer.max_num_epochs = 2
    trainer.num_batches_per_epoch = 2
    trainer.num_val_batches_per_epoch = 2
    trainer.initialize(True)
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
        trainer_class_name=nnUNetTrainerV2Weighted.__name__,
    )


@pytest.mark.parametrize("use_rand_wmap", (False, True))
def test_weighted_DC_and_CE_loss(use_rand_wmap: bool):
    torch.manual_seed(1234)
    kwargs = dict(
        soft_dice_kwargs={"batch_dice": False, "smooth": 1e-5, "do_bg": False},
        aggregate="sum",
        weight_ce=1.0,
        weight_dice=1.0,
    )

    # don't share ce_kwargs, since the classes can change the ce_kwargs (non-mutable issue with side effects)
    loss_weighted = DC_and_CE_loss_weighted(**kwargs, ce_kwargs={})
    loss_ref = DC_and_CE_loss(**kwargs, ce_kwargs={})
    classes = 3
    Ypred = torch.rand(9, classes, 11, 12, 13)
    Y = torch.randint(low=0, high=classes, size=(9, 1, 11, 12, 13))
    if use_rand_wmap:
        W = torch.rand(9, 1, 11, 12, 13)
    else:
        W = torch.ones(9, 1, 11, 12, 13)

    ref_value = loss_ref(Ypred, Y)
    weighted_value = loss_weighted(Ypred, Y, W)

    assert (ref_value == weighted_value) != use_rand_wmap
