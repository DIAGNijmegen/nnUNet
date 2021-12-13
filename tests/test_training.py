import pytest
from pathlib import Path
import sys

import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as nnpap
import nnunet.experiment_planning.utils as nnepu
import nnunet.run.default_configuration as nndc
from nnunet.paths import default_plans_identifier

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.run.run_training import main
from nnunet.training.network_training.nnUNet_variants.tests.nnUNetTrainerV2_test import (
    nnUNetTrainerV2_test,
)
from utils import (
    is_data_integrity_ok_md5sum,
    is_data_present_md5,
    are_pickle_files_roughly_the_same,
)


RESOURCES_DIR = Path(__file__).parent / "resources"
NNUNET_RAW_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_raw_data"
NNUNET_CROPPED_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_cropped_data"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
NNUNET_REF_OUTPUT_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_results"
HIPPOCAMPUS_TASK_ID = 4
TEST_TRAINER_CLASS_NAME = "nnUNetTrainerV2_test"


def prepare_paths(output_dir: Path):
    for pathsmod in [nnp, nnpap, nnepu, nndc]:
        pathsmod.nnUNet_raw_data = str(NNUNET_RAW_DATA_DIR)
        pathsmod.nnUNet_cropped_data = str(NNUNET_CROPPED_DATA_DIR)
        pathsmod.preprocessing_output_dir = str(NNUNET_PREPROCESSING_DATA_DIR)
        pathsmod.network_training_output_dir = str(output_dir)


@pytest.mark.parametrize("network", ("2d", "3d_fullres"))
@pytest.mark.parametrize("fold", (0,))
def test_nnUNetTrainerV2_train_and_validate(tmp_path: Path, network: str, fold: int):
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
        network, task, TEST_TRAINER_CLASS_NAME, default_plans_identifier
    )
    assert issubclass(trainer_class, nnUNetTrainerV2)
    assert nnUNetTrainerV2_test is trainer_class
    trainer = trainer_class(
        plans_file,
        fold,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )
    assert trainer.max_num_epochs == 2
    assert trainer.num_batches_per_epoch == 2
    assert trainer.num_val_batches_per_epoch == 2
    trainer.initialize(True)
    trainer.run_training()
    trainer.network.eval()
    trainer.validate(
        save_softmax=False,
        validation_folder_name="validation_raw",
        run_postprocessing_on_folds=True,
        overwrite=True,
    )
    check_expected_training_output(check_dir=tmp_path, network=network)


@pytest.mark.parametrize("network", ("2d", "3d_fullres"))
@pytest.mark.parametrize("fold", (0,))
def test_train_cli(tmp_path: Path, network: str, fold: int):
    prepare_paths(output_dir=tmp_path)
    # network, network_trainer, task, fold
    sys.argv = [
        "",
        network,
        TEST_TRAINER_CLASS_NAME,
        str(HIPPOCAMPUS_TASK_ID),
        str(fold),
    ]
    main()
    check_expected_training_output(check_dir=tmp_path, network=network)


def check_expected_training_output(check_dir: Path, network: str, fold: int = 0):
    check_dir_sub = (
        check_dir
        / network
        / "Task004_Hippocampus"
        / (TEST_TRAINER_CLASS_NAME + "__nnUNetPlansv2.1")
    )
    # these files should be exactly the same...
    assert is_data_integrity_ok_md5sum(
        workdir=check_dir, md5file=NNUNET_REF_OUTPUT_DIR / (network + ".md5")
    )
    # these files can differ in contents, but should be present...
    assert is_data_present_md5(
        workdir=check_dir, md5file=NNUNET_REF_OUTPUT_DIR / (network + "_other.md5")
    )
    # training log will change datetime in filename...
    check_dir_fold = check_dir_sub / f"fold_{fold}"
    assert len(list(check_dir_fold.glob("training_log_*.txt"))) == 1
    # the plans files are just copied from the preprocessed data dir and should be roughly the same...
    plan_tag = "2D" if network == "2d" else "3D"
    ref_filepath = (
        NNUNET_PREPROCESSING_DATA_DIR
        / "Task004_Hippocampus"
        / f"nnUNetPlansv2.1_plans_{plan_tag}.pkl"
    )
    assert are_pickle_files_roughly_the_same(
        filepath=check_dir_sub / "plans.pkl", ref_filepath=ref_filepath
    )
