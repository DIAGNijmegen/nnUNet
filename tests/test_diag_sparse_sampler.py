from typing import Optional, Tuple

import pytest

from nnunet.training.data_augmentation.diag.transforms.spatial_transforms import augment_spatial_sparse
from batchgenerators.transforms.spatial_transforms import augment_spatial

import numpy as np


@pytest.fixture
def input_data() -> np.ndarray:
    return np.array(
        [[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,0,0,0,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,1,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,2,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,4,0,0,0,0,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,2,2,0,0,0,0,5,0,0,0,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,2,2,0,0,0,0,0,6,0,0,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,2,2,0,0,0,0,0,0,7,0,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,2,2,0,0,0,0,4,4,4,8,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,2,2,0,0,0,0,4,1,4,4,9,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,3,4,5,0,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,1,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,2,3,0,0,0,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,4,0,0,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,3,0,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
         [0,0,0,0,0,0,0,0,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,0,0,0,0],
        ]
    )

@pytest.fixture
def input_data_3d(input_data: np.ndarray) -> np.ndarray:
    slc = np.expand_dims(input_data, axis=0)
    return np.concatenate([slc for _ in range(20)])


CROP_OUTPUT = np.array(
[[[[7, 0, 5, 5, 5,],
   [4, 8, 5, 5, 5,],
   [4, 4, 9, 5, 5,],
   [3, 4, 5, 0, 5,],
   [4, 4, 5, 5, 1,]]]]
)

CROP_SCALE_OUTPUT = np.array(
[[[[7, 0, 5, 5, 5,],
   [4, 8, 5, 5, 5,],
   [4, 4, 9, 5, 5,],
   [3, 4, 5, 0, 5,],
   [4, 4, 5, 5, 1,]]]]
)

CROP_ROTA_OUTPUT = np.array(
[[[[5, 5, 4, 3, 1],
   [5, 5, 4, 4, 4],
   [5, 0, 9, 8, 4],
   [5, 5, 5, 5, 0],
   [5, 5, 5, 5, 5]]]])

CROP_ELASTIC_OUTPUT = np.array(
[[[[3, 4, 5, 0, 5,],
   [4, 4, 5, 5, 1,],
   [4, 4, 5, 5, 5,],
   [4, 4, 5, 5, 5,],
   [4, 4, 5, 5, 5,],]]])

CROP_ELASTIC_ROTA_OUTPUT = np.array(
[[[[4, 4, 4, 4, 5],
   [4, 1, 3, 4, 4],
   [4, 4, 4, 4, 4],
   [0, 4, 4, 4, 4],
   [3, 3, 3, 4, 4]]]])

CROP_SCALE_ROTA_OUTPUT = np.array(
[[[[5, 5, 4, 3, 1,],
   [5, 5, 4, 4, 4,],
   [0, 0, 9, 8, 8,],
   [5, 5, 5, 5, 0,],
   [5, 5, 5, 5, 5,]]]]
)

CROP_SCALE_ELASTIC_OUTPUT = np.array(
[[[[3, 4, 5, 0, 0,],
   [4, 4, 5, 5, 5,],
   [4, 4, 5, 5, 5,],
   [4, 4, 5, 5, 5,],
   [4, 4, 5, 5, 5,]]]]
)

CROP_SCALED_DEFORM_ROTA_OUTPUT = np.array(
[[[[4, 4, 4, 4, 5,],
   [4, 4, 3, 4, 4,],
   [1, 4, 4, 4, 4,],
   [4, 4, 4, 4, 4,],
   [3, 4, 4, 4, 4,]]]]
)

CROP_OUTPUT_3D = np.array(
[[[[[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]]]]])

CROP_ROTA_OUTPUT_3D = np.array(
[[[[[0, 5, 5,],
    [8, 9, 9,],
    [4, 4, 5,]],
   [[5, 5, 5,],
    [4, 9, 5,],
    [4, 4, 5,]],
   [[5, 5, 5,],
    [9, 9, 0,],
    [4, 5, 5,]]]]])

CROP_ELASTIC_OUTPUT_3D = np.array(
[[[[[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]]]]])

CROP_ELASTIC_ROTA_OUTPUT_3D = np.array(
[[[[[5, 5, 5,],
    [8, 5, 5,],
    [8, 9, 5,]],
   [[8, 9, 5,],
    [4, 9, 5,],
    [4, 9, 0,]],
   [[4, 4, 5,],
    [4, 4, 5,],
    [3, 4, 5,]]]]])

CROP_SCALE_OUTPUT_3D = np.array(
[[[[[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]]]]])

CROP_SCALE_ROTA_OUTPUT_3D = np.array(
[[[[[0, 5, 5,],
    [8, 4, 9,],
    [4, 4, 5,]],
   [[5, 5, 5,],
    [4, 9, 5,],
    [4, 4, 5,]],
   [[5, 5, 5,],
    [9, 5, 0,],
    [4, 5, 5,]]]]])

CROP_SCALE_ELASTIC_OUTPUT_3D = np.array(
[[[[[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]],
   [[8, 5, 5,],
    [4, 9, 5,],
    [4, 5, 0,]]]]])

CROP_SCALED_DEFORM_ROTA_OUTPUT_3D = np.array(
[[[[[5, 5, 5,],
    [8, 5, 5,],
    [8, 9, 5,]],
   [[8, 9, 5,],
    [4, 9, 5,],
    [4, 9, 0,]],
   [[4, 9, 5,],
    [4, 4, 5,],
    [3, 4, 5,]]]]])


@pytest.mark.parametrize("use_3d", (False, True))
@pytest.mark.parametrize("random_crop", (False, True))  # random_crop only matters when use_sparse_mask = False
@pytest.mark.parametrize(
    "use_sparse_mask, do_scale, do_elastic_deform, do_rotation, expected_output",
    (
        (False, False, False, False, None),
        (False, False, False, True, None),
        (False, False, True, False, None),
        (False, False, True, True, None),
        (False, True, False, False, None),
        (False, True, False, True, None),
        (False, True, True, False, None),
        (False, True, True, True, None),

        (True, False, False, False, (CROP_OUTPUT, CROP_OUTPUT_3D)),
        (True, False, False, True, (CROP_ROTA_OUTPUT, CROP_ROTA_OUTPUT_3D)),
        (True, False, True, False, (CROP_ELASTIC_OUTPUT, CROP_ELASTIC_OUTPUT_3D)),
        (True, False, True, True, (CROP_ELASTIC_ROTA_OUTPUT, CROP_ELASTIC_ROTA_OUTPUT_3D)),
        (True, True, False, False, (CROP_SCALE_OUTPUT, CROP_SCALE_OUTPUT_3D)),
        (True, True, False, True, (CROP_SCALE_ROTA_OUTPUT, CROP_SCALE_ROTA_OUTPUT_3D)),
        (True, True, True, False, (CROP_SCALE_ELASTIC_OUTPUT, CROP_SCALE_ELASTIC_OUTPUT_3D)),
        (True, True, True, True, (CROP_SCALED_DEFORM_ROTA_OUTPUT, CROP_SCALED_DEFORM_ROTA_OUTPUT_3D)),
    ),
)
def test_compare_augment_spatial_vs_diag_spatial_sparse(
    input_data: np.ndarray,
    input_data_3d: np.ndarray,
    use_sparse_mask: bool,
    random_crop: bool,
    do_scale: bool,
    do_elastic_deform: bool,
    do_rotation: bool,
    expected_output: Optional[Tuple[np.ndarray, np.ndarray]],
    use_3d: bool,
    sample_mask_label: int = 9,
):
    data = np.expand_dims(np.expand_dims(input_data_3d if use_3d else input_data, 0), 0)
    seg = np.copy(data)
    sample_mask = seg == sample_mask_label
    patch_size = (3, 3, 3) if use_3d else (5, 5)
    patch_center_dist_from_border = 4

    kwargs = dict(
        data=data,
        seg=seg,
        patch_size=patch_size,
        patch_center_dist_from_border=patch_center_dist_from_border,
        random_crop=random_crop,
        do_scale=do_scale,
        do_rotation=do_rotation,
        do_elastic_deform=do_elastic_deform,
        p_el_per_sample=1.0,
        p_rot_per_sample=1.0,
        p_scale_per_sample=1.0,
    )

    np.random.seed(seed=0)
    augdata, augseg = augment_spatial_sparse(
        sample_mask=sample_mask,
        sample_on_sparse_mask_only=use_sparse_mask,
        **kwargs
    )

    if not use_sparse_mask:
        np.random.seed(seed=0)
        augdata_ref, augseg_ref = augment_spatial(
            **kwargs,
        )
        assert np.array_equal(augdata, augdata_ref)
        assert np.array_equal(augseg, augseg_ref)
    else:
        # Note!!! cannot ensure that label used for masking is present in the output augmented segmentation
        # This is because sampling at edges is possible. Hence, following assert would fail in some cases:
        # assert sample_mask_label in augseg
        # To ensure this doesn't give problems, put sufficient annotations for each scan and not just a few voxels.
        assert np.array_equal(augseg, expected_output[1 if use_3d else 0])
    assert augdata.shape[2:] == patch_size
    assert augseg.shape[2:] == patch_size
