"""
DEPRECATED SHIM — phase4.api.app moved to nhpc_qa.api.app.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase4.api.app
    new:  nhpc serve
"""
import warnings as _w
_w.warn("phase4.api.app has moved to nhpc_qa.api.app; use `nhpc serve`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.api.app import *          # noqa: F401,F403
from nhpc_qa.api.app import main       # noqa: F401
