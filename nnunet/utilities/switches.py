"""
This module contains variables that indicate which additional DIAG features should be
active. These switches are set based on environment variables. These variables should
have a boolean-like value such as 0 for "off" and 1 for "on". The default setting is
"off".

The following switches are available:

- DIAG_NNUNET_ALT_RESAMPLING
    Activates an alternative resampling strategy for resampling of the softmax output
    to the original image resolution. The original implementation can require a lot of
    memory, the alternative implementation avoids the use of subprocesses and uses a
    somewhat more memory efficient resampling strategy. Note that this strategy might be
    slower though.
"""

import os


def switch(environment_variable):
    return bool(os.environ.get(environment_variable, default=False))


use_alt_resampling = switch("DIAG_NNUNET_ALT_RESAMPLING")
