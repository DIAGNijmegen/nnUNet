"""
When copying files from a compute node to a storage server on DIAG's cluster, using the
standard shutil copy() and copytree() functions leads to errors. These functions copy
files but also try to copy metadata such as permissions, and our storage servers to not
support that. This module defines replacement functions.
"""

import os
import shutil


def copyfile(src, dst, **kwargs):
    """Similar to shutil.copyfile but accepts a directory as input for dst"""
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    return shutil.copyfile(src, dst, **kwargs)


def copytree(src, dst, ignore=None):
    """Similar to shutil.copytree but makes sure that copyfile is used for copying"""
    try:
        shutil.copytree(src, dst,
                        ignore=ignore,
                        symlinks=False,
                        ignore_dangling_symlinks=True,
                        copy_function=copyfile)
    except shutil.Error as e:
        non_permission_errors = []
        for error in e.args[0]:
            msg = error[2] if isinstance(error, tuple) else error
            if 'Operation not permitted' not in msg:
                non_permission_errors.append(error)

        if len(non_permission_errors) > 0:
            raise shutil.Error(non_permission_errors)

    return dst
