import pickle

from typing import Dict
from pathlib import Path

import subprocess as sp

import numpy as np


def is_data_integrity_ok_md5sum(workdir: Path, md5file: Path) -> bool:
    result = sp.run(["md5sum", "--check", "--quiet", str(md5file)], cwd=str(workdir))
    return result.returncode == 0


def is_data_present_md5(workdir: Path, md5file: Path) -> bool:
    with open(md5file, "r") as f:
        data = f.readlines()
    for line in data:
        filepath = Path(line.split("  ")[1].strip())
        if not (workdir / filepath).is_file():
            print(f"{filepath} does not exist in {workdir}")
            return False
        else:
            print(f"{filepath}: OK")
    return True


def are_plan_files_roughly_the_same(filepath: Path, ref_filepath: Path) -> bool:
    assert filepath.is_file()
    with open(str(filepath), "rb") as f:
        obj = pickle.load(f)
    with open(str(ref_filepath), "rb") as f2:
        obj2 = pickle.load(f2)
    for k, v in obj2.items():
        if k not in obj:
            return False
        if k in ["list_of_npz_files", "preprocessed_data_folder"]:
            pass
        elif k in ["dataset_properties", "plans_per_stage"]:
            if not compare_dicts(a=obj[k], b=v):
                return False
        elif k == "original_spacings":
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
