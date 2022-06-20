import os
import shutil
import tarfile
import sys
from pathlib import Path
from contextlib import contextmanager

import nnunet.inference.pretrained_models.download_pretrained_model
import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as nnpap
import nnunet.experiment_planning.utils as nnepu
from nnunet.experiment_planning.nnUNet_plan_and_preprocess import (
    main as plan_and_preprocess_main,
)
from nnunet.experiment_planning.nnUNet_convert_decathlon_task import (
    main as convert_main,
)
from nnunet.inference.pretrained_models.download_pretrained_model import (
    download_and_install_pretrained_model_by_name,
)
from nnunet.utilities.diag.generate_unet_weightmaps import generate_unet_weightmaps, get_classes_from_dataset_file

RESOURCES_DIR = Path(__file__).parent
PREPROCESSED_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
TEST_INPUT_DATA_DIR = RESOURCES_DIR / "input_data"
DECATHLON_HIPPOCAMPUS_FILE = TEST_INPUT_DATA_DIR / "Task04_Hippocampus.tar"
DECATHLON_TASK04_HIPPOCAMPUS_DIR = TEST_INPUT_DATA_DIR / "Task04_Hippocampus"
NNUNET_RAW_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_raw_data"
NNUNET_CROPPED_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_cropped_data"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
HIPPOCAMPUS_TASK = "Task004_Hippocampus"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def download_pretrained_models():
    print("  # downloading pretrained models")
    nnunet.inference.pretrained_models.download_pretrained_model.network_training_output_dir = str(
        RESOURCES_DIR / "pretrained" / HIPPOCAMPUS_TASK
    )
    with suppress_stdout():
        download_and_install_pretrained_model_by_name(HIPPOCAMPUS_TASK)


def unpack_decathlon_hippocampus_dataset():
    print("  # unpacking decathlon hippocampus dataset")
    with tarfile.TarFile(str(DECATHLON_HIPPOCAMPUS_FILE), "r") as f:
        f.extractall(TEST_INPUT_DATA_DIR)


def create_dummy_task004_input_data():
    print("  # creating dummy task004 input data")
    SRC_DIR = TEST_INPUT_DATA_DIR / "Task04_Hippocampus" / "imagesTs"
    TEST_DIR = TEST_INPUT_DATA_DIR / HIPPOCAMPUS_TASK
    TEST_IMGS_DIR = TEST_DIR / "imagesTs"
    TEST_DIR.mkdir(exist_ok=True)
    TEST_IMGS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(
        src=str(SRC_DIR / "hippocampus_002.nii.gz"),
        dst=str(TEST_IMGS_DIR / "hippocampus_002_0000.nii.gz"),
    )
    shutil.copyfile(
        src=str(SRC_DIR / "hippocampus_005.nii.gz"),
        dst=str(TEST_IMGS_DIR / "hippocampus_005_0000.nii.gz"),
    )


def convert_decathlon_dataset_to_nnunet_format():
    print("  # converting decathlon dataset to nnunet format")
    for pathsmod in [nnepu]:
        pathsmod.nnUNet_raw_data = str(NNUNET_RAW_DATA_DIR)
    sys.argv = ["", "-i", str(DECATHLON_TASK04_HIPPOCAMPUS_DIR)]
    with suppress_stdout():
        convert_main()


def generate_cropped_and_preprocessed_files():
    print("  # generating cropped and preprocessed files")
    for pathsmod in [nnp, nnpap, nnepu]:
        pathsmod.nnUNet_raw_data = str(NNUNET_RAW_DATA_DIR)
        pathsmod.nnUNet_cropped_data = str(NNUNET_CROPPED_DATA_DIR)
        pathsmod.preprocessing_output_dir = str(NNUNET_PREPROCESSING_DATA_DIR)
    sys.argv = ["", "-t", "4", "--verify_dataset_integrity"]
    with suppress_stdout():
        plan_and_preprocess_main()


def bootstrap_diag_weightmaps():
    print("  # bootstrapping diag weightmaps for preprocessed files")
    input_dir = NNUNET_PREPROCESSING_DATA_DIR / HIPPOCAMPUS_TASK / "nnUNetData_plans_v2.1_stage0"
    with suppress_stdout():
        classes = get_classes_from_dataset_file(input_dir=input_dir, background_label=0)
        generate_unet_weightmaps(
            input_dir=input_dir,
            output_dir=input_dir,
            matching_pattern="*.npz",
            sigma=5.0,
            w_0=10.0,
            classes=classes,
        )


def bootstrap():
    download_pretrained_models()
    unpack_decathlon_hippocampus_dataset()
    create_dummy_task004_input_data()
    convert_decathlon_dataset_to_nnunet_format()
    generate_cropped_and_preprocessed_files()
    bootstrap_diag_weightmaps()


if __name__ == "__main__":
    bootstrap()
