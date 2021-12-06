import sys
import subprocess as sp
from pathlib import Path


RESOURCE_DIR = Path(__file__).parent
PREPROCESSED_DATA_DIR = RESOURCE_DIR / "nnUNet" / "nnUNet_preprocessed_data"
SOURCE_TAR_FILE = RESOURCE_DIR / "input_data" / "Task04_Hippocampus.tar"
SOURCE_TAR_FILE_MD5 = b"9d24dba78a72977dbd1d2e110310f31b"


def print_download_instructions():
    print("Download the original file here: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2")
    print(f"And place it in the following directory: {SOURCE_TAR_FILE.parent}")


def check_source_file_ok_or_exit():
    if not SOURCE_TAR_FILE.is_file():
        print(f"Missing the source TAR file to bootstrap from: {SOURCE_TAR_FILE.name}")
        print_download_instructions()
        sys.exit(2)
    result = sp.run(["md5sum", str(SOURCE_TAR_FILE)], cwd=str(SOURCE_TAR_FILE.parent), stdout=sp.PIPE)
    if result.stdout[:len(SOURCE_TAR_FILE_MD5)] != SOURCE_TAR_FILE_MD5:
        print(f"Found the source TAR file ({SOURCE_TAR_FILE.name}), but it does not match its MD5 hash ({SOURCE_TAR_FILE_MD5})!")
        print_download_instructions()
        sys.exit(2)


def is_data_integrity_ok_md5sum(workdir: Path, md5file: Path) -> bool:
    result = sp.run(["md5sum", "--check", str(md5file)], cwd=str(workdir), stdout=sp.DEVNULL)
    return result.returncode == 0


def check_if_ok_or_exit(workdir: Path, md5file: Path):
    if not is_data_integrity_ok_md5sum(workdir=workdir, md5file=md5file):
        print(f"MD5 Checksum for {md5file} failed, exiting...")
        sys.exit(1)


def check_if_file_exists_or_exit(filepath: Path):
    if not filepath.is_file():
        print(f"{filepath} does not exist, exiting...")
        sys.exit(1)


def check_integrity():
    check_source_file_ok_or_exit()
    check_if_ok_or_exit(
        workdir=RESOURCE_DIR / "input_data",
        md5file=RESOURCE_DIR / "input_data" / "Task04_Hippocampus.md5",
    )
    check_if_ok_or_exit(
        workdir=RESOURCE_DIR / "input_data",
        md5file=RESOURCE_DIR / "input_data" / "Task004_Hippocampus.md5",
    )
    check_if_ok_or_exit(
        workdir=RESOURCE_DIR / "pretrained",
        md5file=RESOURCE_DIR / "pretrained" / "Task004_Hippocampus.md5",
    )
    check_if_ok_or_exit(
        workdir=RESOURCE_DIR / "nnUNet" / "nnUNet_cropped_data",
        md5file=RESOURCE_DIR
        / "nnUNet"
        / "nnUNet_cropped_data"
        / "Task004_Hippocampus.md5",
    )
    check_if_ok_or_exit(
        workdir=PREPROCESSED_DATA_DIR,
        md5file=PREPROCESSED_DATA_DIR / "Task004_Hippocampus.md5",
    )
    check_if_file_exists_or_exit(
        filepath=PREPROCESSED_DATA_DIR
        / "Task004_Hippocampus"
        / "nnUNetPlansv2.1_plans_2D.pkl"
    )
    check_if_file_exists_or_exit(
        filepath=PREPROCESSED_DATA_DIR
        / "Task004_Hippocampus"
        / "nnUNetPlansv2.1_plans_3D.pkl"
    )


if __name__ == "__main__":
    check_integrity()
