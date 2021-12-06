from pathlib import Path

import pytest

import nnunet.inference.pretrained_models.download_pretrained_model
from nnunet.inference.pretrained_models.download_pretrained_model import (
    download_and_install_pretrained_model_by_name,
    print_available_pretrained_models,
    get_available_models,
)
from utils import is_data_integrity_ok_md5sum


TASK004_HIPPOCAMPUS_PRETRAINED_DIR = (
    Path(__file__).parent / "resources" / "pretrained" / "Task004_Hippocampus"
)
TASK004_HIPPOCAMPUS_MANIFEST_DIR = Path(__file__).parent / "resources" / "pretrained"

AVAILABLE_MODELS = [
    "Task001_BrainTumour",
    "Task002_Heart",
    "Task003_Liver",
    "Task004_Hippocampus",
    "Task005_Prostate",
    "Task006_Lung",
    "Task007_Pancreas",
    "Task008_HepaticVessel",
    "Task009_Spleen",
    "Task010_Colon",
    "Task017_AbdominalOrganSegmentation",
    "Task024_Promise",
    "Task027_ACDC",
    "Task029_LiTS",
    "Task035_ISBILesionSegmentation",
    "Task038_CHAOS_Task_3_5_Variant2",
    "Task048_KiTS_clean",
    "Task055_SegTHOR",
    "Task061_CREMI",
    "Task075_Fluo_C3DH_A549_ManAndSim",
    "Task076_Fluo_N3DH_SIM",
    "Task082_BraTS2020",
    "Task089_Fluo-N2DH-SIM_thickborder_time",
    "Task114_heart_MNMs",
    "Task115_COVIDSegChallenge",
    "Task135_KiTS2021",
]


@pytest.mark.parametrize("taskname", ("Task004_Hippocampus",))
def test_nnunet_download_pretrained(tmp_path: Path, taskname: str):
    nnunet.inference.pretrained_models.download_pretrained_model.network_training_output_dir = str(
        tmp_path / taskname
    )
    download_and_install_pretrained_model_by_name(taskname=taskname)
    assert is_data_integrity_ok_md5sum(
        workdir=tmp_path, md5file=TASK004_HIPPOCAMPUS_MANIFEST_DIR / (taskname + ".md5")
    )


def test_nnunet_get_available_pretrained_models():
    models = get_available_models()
    assert set(models.keys()) == set(AVAILABLE_MODELS)


def test_nnunet_print_available_pretrained_models():
    print_available_pretrained_models()
