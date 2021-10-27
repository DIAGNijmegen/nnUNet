import os


def _make_switch(environment_variable):
    def switch() -> bool:
        return bool(os.environ.get(environment_variable, default=False))
    return switch


use_alt_resampling = _make_switch("DIAG_NNUNET_ALT_RESAMPLING")
