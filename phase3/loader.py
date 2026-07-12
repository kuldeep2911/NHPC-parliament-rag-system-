"""
DEPRECATED SHIM — phase3.loader moved to nhpc_qa.pipeline.index.loader.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase3.loader
    new:  nhpc run --stages index
"""
import warnings as _w
_w.warn("phase3.loader has moved to nhpc_qa.pipeline.index.loader; use `nhpc run --stages index`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.pipeline.index.loader import *          # noqa: F401,F403
from nhpc_qa.pipeline.index.loader import main       # noqa: F401
