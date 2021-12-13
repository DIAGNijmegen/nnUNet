import pickle

from typing import Dict, Optional
from pathlib import Path

import subprocess as sp

import numpy as np


def is_data_integrity_ok_md5sum(workdir: Path, md5file: Path) -> bool:
    result = sp.run(["md5sum", "--check", "--quiet", str(md5file)], cwd=str(workdir))
    return result.returncode == 0


def is_data_present_md5(workdir: Path, md5file: Path, ref_pickle_path: Optional[Path] = None) -> bool:
    with open(md5file, "r") as f:
        data = f.readlines()
    for line in data:
        filepath = Path(line.split("  ")[1].strip())
        if not (workdir / filepath).is_file():
            print(f"{filepath} does not exist in {workdir}")
            return False
        elif ref_pickle_path is not None and filepath.suffix == ".pkl" and not are_pickle_files_roughly_the_same(
            filepath=workdir / filepath, ref_filepath=ref_pickle_path / filepath
        ):
            print(f"Pickle file {filepath} in {workdir} does not roughly match that in {ref_pickle_path}")
            return False
    return True


def are_pickle_files_roughly_the_same(filepath: Path, ref_filepath: Path) -> bool:
    assert filepath.is_file()
    with open(str(filepath), "rb") as f:
        obj = pickle.load(f)
    with open(str(ref_filepath), "rb") as f2:
        obj2 = pickle.load(f2)
    for k, v in obj2.items():
        if k not in obj:
            return False
        if k in ["list_of_npz_files", "list_of_data_files"]:
            if not (isinstance(v, list) and all([isinstance(e, str) for e in v])):
                return False
        elif k in ["preprocessed_data_folder", "seg_file"]:
            if not isinstance(v, str):
                return False
        elif k in ["dataset_properties", "plans_per_stage", "use_nonzero_mask_for_norm", "class_locations", "modalities", "normalization_schemes"]:
            if not compare_dicts(a=obj[k], b=v):
                return False
        elif k in ["original_spacings", "original_size_of_raw_data", "original_spacing", "crop_bbox", "classes", "itk_origin", "itk_spacing", "itk_direction", "size_after_cropping", "size_after_resampling", "spacing_after_resampling", "original_sizes"]:
            if not np.all([a == b for a, b in zip(obj[k], v)]):
                return False
        else:
            if obj[k] != v:
                return False
    return True


def compare_dicts(a: Dict, b: Dict) -> bool:
    for key, value in b.items():
        if key not in a:
            return False
        if isinstance(value, dict):
            if not compare_dicts(a[key], value):
                return False
        elif (
            isinstance(value, (list, tuple)) and isinstance(value[0], np.ndarray)
        ) or isinstance(value, np.ndarray):
            if not np.all([a == b for a, b in zip(a[key], value)]):
                return False
        else:
            if a[key] != value:
                return False
    return True
