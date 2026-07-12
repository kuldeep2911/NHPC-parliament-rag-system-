"""
DEPRECATED SHIM — phase3.inspect_db moved to nhpc_qa.scripts.inspect_db.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase3.inspect_db
    new:  nhpc inspect
"""
import warnings as _w
_w.warn("phase3.inspect_db has moved to nhpc_qa.scripts.inspect_db; use `nhpc inspect`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.scripts.inspect_db import *          # noqa: F401,F403
from nhpc_qa.scripts.inspect_db import main       # noqa: F401
