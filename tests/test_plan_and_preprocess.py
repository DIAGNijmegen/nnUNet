from pathlib import Path

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

from utils import is_data_integrity_ok_md5sum, is_data_present_md5


RESOURCES_DIR = Path(__file__).parent / "resources"
NNUNET_RAW_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_raw_data"
NNUNET_CROPPED_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_cropped_data"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
DECATHLON_TASK04_HIPPOCAMPUS_DIR = RESOURCES_DIR / "input_data" / "Task04_Hippocampus"


def test_convert_decathlon_dataset(tmp_path: Path):
    for pathsmod in [nnepu]:
        pathsmod.nnUNet_raw_data = str(tmp_path)
    sys.argv = ["", "-i", str(DECATHLON_TASK04_HIPPOCAMPUS_DIR)]
    convert_main()
    assert is_data_integrity_ok_md5sum(
        workdir=tmp_path, md5file=NNUNET_RAW_DATA_DIR / "Task004_Hippocampus.md5"
    )


def test_plan_and_preprocess(tmp_path: Path):
    TMP_CROPPED_DIR = tmp_path / "cropped"
    TMP_PREPROCESSING_DIR = tmp_path / "preprocessing"
    for path_dir in [TMP_CROPPED_DIR, TMP_PREPROCESSING_DIR]:
        path_dir.mkdir()
    for pathsmod in [nnp, nnpap, nnepu]:
        pathsmod.nnUNet_raw_data = str(NNUNET_RAW_DATA_DIR)
        pathsmod.nnUNet_cropped_data = str(TMP_CROPPED_DIR)
        pathsmod.preprocessing_output_dir = str(TMP_PREPROCESSING_DIR)
    sys.argv = ["", "-t", "4", "--verify_dataset_integrity"]
    plan_and_preprocess_main()
    # assert generated files are matching the references...
    assert is_data_integrity_ok_md5sum(
        workdir=TMP_CROPPED_DIR,
        md5file=NNUNET_CROPPED_DATA_DIR / "Task004_Hippocampus.md5",
    )
    assert is_data_present_md5(
        workdir=TMP_CROPPED_DIR,
        md5file=NNUNET_CROPPED_DATA_DIR / "Task004_Hippocampus_other.md5",
        ref_pickle_path=NNUNET_CROPPED_DATA_DIR,
    )
    assert is_data_integrity_ok_md5sum(
        workdir=TMP_PREPROCESSING_DIR,
        md5file=NNUNET_PREPROCESSING_DATA_DIR / "Task004_Hippocampus.md5",
    )
    assert is_data_present_md5(
        workdir=TMP_PREPROCESSING_DIR,
        md5file=NNUNET_PREPROCESSING_DATA_DIR / "Task004_Hippocampus_other.md5",
        ref_pickle_path=NNUNET_PREPROCESSING_DATA_DIR,
    )
