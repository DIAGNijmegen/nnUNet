from typing import Dict

from pathlib import Path

import numpy as np
import pickle
import sys

import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as nnpap
import nnunet.experiment_planning.utils as nnepu
from nnunet.experiment_planning.nnUNet_plan_and_preprocess import (
    main as plan_and_preprocess_main,
)
from nnunet.experiment_planning.nnUNet_convert_decathlon_task import (
    main as convert_main,
)

from utils import is_data_integrity_ok_md5sum

RESOURCES_DIR = Path(__file__).parent / "resources"

NNUNET_RAW_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_raw_data"
NNUNET_CROPPED_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_cropped_data"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
# NNUNET_REF_OUTPUT_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_results"
DECATHLON_TASK04_HIPPOCAMPUS_DIR = RESOURCES_DIR / "input_data" / "Task04_Hippocampus"


def test_convert_decathlon_dataset(tmp_path: Path):
    for pathsmod in [nnepu]:
        pathsmod.nnUNet_raw_data = str(tmp_path)
    sys.argv = ["", "-i", str(DECATHLON_TASK04_HIPPOCAMPUS_DIR)]
    convert_main()
    assert is_data_integrity_ok_md5sum(
        workdir=tmp_path, md5file=NNUNET_RAW_DATA_DIR / "Task004_Hippocampus.md5"
    )


def check_pickle_files_roughly_the_same(
    pickle_file: Path, tmp_dir: Path, ref_dir: Path
):
    assert (tmp_dir / pickle_file).is_file()
    with open(str(tmp_dir / pickle_file), "rb") as f:
        obj = pickle.load(f)
    with open(str(ref_dir / pickle_file), "rb") as f2:
        obj2 = pickle.load(f2)
    for k, v in obj2.items():
        assert k in obj
        if k in ["list_of_npz_files", "preprocessed_data_folder"]:
            assert obj[k] != v
        elif k in ["dataset_properties", "plans_per_stage"]:
            assert compare_dicts(a=obj[k], b=v)
        elif k == "original_spacings":
            assert np.all([a == b for a, b in zip(obj[k], v)])
        else:
            assert obj[k] == v


def compare_dicts(a: Dict, b: Dict) -> bool:
    for key, value in b.items():
        if key not in a:
            return False
        if isinstance(value, dict):
            if not compare_dicts(a[key], value):
                return False
        elif (isinstance(value, (list, tuple)) and isinstance(value[0], np.ndarray)) or isinstance(value, np.ndarray):
            if not np.all([a == b for a, b in zip(a[key], value)]):
                return False
        else:
            if a[key] != value:
                return False
    return True


def test_plan_and_preprocess(tmp_path: Path):
    TMP_CROPPED_DIR = tmp_path / "cropped"
    TMP_PREPROCESSING_DIR = tmp_path / "preprocessing"
    for path_dir in [TMP_CROPPED_DIR, TMP_PREPROCESSING_DIR]:
        path_dir.mkdir()
    for pathsmod in [nnp, nnpap, nnepu]:
        pathsmod.nnUNet_raw_data = str(NNUNET_RAW_DATA_DIR)
        pathsmod.nnUNet_cropped_data = str(TMP_CROPPED_DIR)
        pathsmod.preprocessing_output_dir = str(TMP_PREPROCESSING_DIR)
        # pathsmod.network_training_output_dir = str(NNUNET_REF_OUTPUT_DIR)
    sys.argv = ["", "-t", "4", "--verify_dataset_integrity"]
    plan_and_preprocess_main()
    # assert generated files are matching the references...
    assert is_data_integrity_ok_md5sum(
        workdir=TMP_CROPPED_DIR,
        md5file=NNUNET_CROPPED_DATA_DIR / "Task004_Hippocampus.md5",
    )
    assert is_data_integrity_ok_md5sum(
        workdir=TMP_PREPROCESSING_DIR,
        md5file=NNUNET_PREPROCESSING_DATA_DIR / "Task004_Hippocampus.md5",
    )
    # Test the two differing plan files independently
    DIFFERING_FILES = ("nnUNetPlansv2.1_plans_2D.pkl", "nnUNetPlansv2.1_plans_3D.pkl")
    for fn in DIFFERING_FILES:
        check_pickle_files_roughly_the_same(
            pickle_file=Path("Task004_Hippocampus") / fn,
            tmp_dir=TMP_PREPROCESSING_DIR,
            ref_dir=NNUNET_PREPROCESSING_DATA_DIR,
        )
