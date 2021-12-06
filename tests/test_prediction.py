from typing import Optional, Tuple

import SimpleITK
import numpy as np
import pytest
import sys
from pathlib import Path

import nnunet.inference.predict_simple
from nnunet.inference.predict_simple import main

RESOURCES_DIR = Path(__file__).parent / "resources"
TASK004_HIPPOCAMPUS_PRETRAINED_DIR = RESOURCES_DIR / "pretrained" / "Task004_Hippocampus"
TEST_INPUT_FOLDER = RESOURCES_DIR / "input_data" / "Task004_Hippocampus" / "imagesTs"
TEST_REF_FOLDER = RESOURCES_DIR / "results"


@pytest.mark.parametrize("model", ("2d", "3d_fullres",))
@pytest.mark.parametrize("folds", (None, (0, 1, 2, 3, 4), (0,),))
@pytest.mark.parametrize("disable_tta", (False, True,))
@pytest.mark.parametrize("use_overlap", (False, True,))
def test_nnunet_inference_predict_simple(tmp_path: Path, model: str, folds: Optional[Tuple[int, ...]], disable_tta: bool, use_overlap: bool):
    fold_dir = f"folds_{folds[0]}" if folds is not None and len(folds) == 1 else "folds_all"
    tta_dir = "notta" if disable_tta else "tta"
    ref_dir = TEST_REF_FOLDER / tta_dir / fold_dir / model
    # set the output_dir by setting the module's variable (environment variables are circumvented this way)
    nnunet.inference.predict_simple.network_training_output_dir = str(TASK004_HIPPOCAMPUS_PRETRAINED_DIR)
    # simulate passing arguments to main() using sys.argv
    sys.argv = ["", "-i", str(TEST_INPUT_FOLDER), "-o", str(tmp_path), "-t", "Task004_Hippocampus", "-m", model]
    if folds is not None:
        sys.argv.extend(["-f"] + list(map(str, folds)))
    if disable_tta:
        sys.argv.append("--disable_tta")
    sys.argv.extend(["--step_size", "1" if not use_overlap else "0.5"])
    main()
    assert (tmp_path / "plans.pkl").is_file()
    assert (tmp_path / "postprocessing.json").is_file()
    for expected_predict_file in ref_dir.glob("*.nii.gz"):
        produced_output_file = (tmp_path / expected_predict_file.name)
        assert produced_output_file.is_file()
        produced_output = SimpleITK.ReadImage(str(produced_output_file))
        expected_output = SimpleITK.ReadImage(str(expected_predict_file))
        assert np.sum(SimpleITK.GetArrayFromImage(produced_output) != SimpleITK.GetArrayFromImage(expected_output)) < 5
