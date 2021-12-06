from pathlib import Path

import subprocess as sp


def is_data_integrity_ok_md5sum(workdir: Path, md5file: Path) -> bool:
    result = sp.run(["md5sum", "--check", str(md5file)], cwd=str(workdir))
    return result.returncode == 0
