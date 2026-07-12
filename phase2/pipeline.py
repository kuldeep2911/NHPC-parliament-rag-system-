"""
DEPRECATED SHIM — phase2.pipeline moved to nhpc_qa.pipeline.parse.pipeline.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase2.pipeline
    new:  nhpc run --stages parse
"""
import warnings as _w
_w.warn("phase2.pipeline has moved to nhpc_qa.pipeline.parse.pipeline; use `nhpc run --stages parse`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.pipeline.parse.pipeline import *          # noqa: F401,F403
from nhpc_qa.pipeline.parse.pipeline import main       # noqa: F401
