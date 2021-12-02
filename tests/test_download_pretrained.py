from typing import Dict

from pathlib import Path

import pytest

import nnunet.inference.pretrained_models.download_pretrained_model
from nnunet.inference.pretrained_models.download_pretrained_model import (
    download_and_install_pretrained_model_by_name,
    print_available_pretrained_models,
    get_available_models,
)


TASK004_HIPPOCAMPUS_PRETRAINED_DIR = Path(__file__).parent / "resources" / "pretrained" / "Task004_Hippocampus"
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


def read_manifest_file(file_path: Path) -> Dict[Path, int]:
    with open(file_path, "r") as f:
        data = f.readlines()
    file_dict = {}
    for line in data:
        file_name, file_size_bytes = line.strip().rsplit(" ", 1)
        file_dict[Path(file_name)] = int(file_size_bytes)
    return file_dict


@pytest.mark.parametrize("taskname", ("Task004_Hippocampus",))
def test_nnunet_download_pretrained(tmp_path: Path, taskname: str):
    manifest = read_manifest_file(TASK004_HIPPOCAMPUS_MANIFEST_DIR / (taskname + ".manifest"))
    nnunet.inference.pretrained_models.download_pretrained_model.network_training_output_dir = str(tmp_path)
    download_and_install_pretrained_model_by_name(taskname=taskname)

    assert any(manifest.keys())
    for file_path, file_size in manifest.items():
        assert (tmp_path / file_path).is_file()
        assert (tmp_path / file_path).stat().st_size == file_size


def test_nnunet_get_available_pretrained_models():
    models = get_available_models()
    assert set(models.keys()) == set(AVAILABLE_MODELS)


def test_nnunet_print_available_pretrained_models():
    print_available_pretrained_models()
