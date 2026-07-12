"""
DEPRECATED SHIM — phase3.migrate moved to nhpc_qa.core.db.migrate.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase3.migrate
    new:  nhpc migrate
"""
import warnings as _w
_w.warn("phase3.migrate has moved to nhpc_qa.core.db.migrate; use `nhpc migrate`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.core.db.migrate import *          # noqa: F401,F403
from nhpc_qa.core.db.migrate import main       # noqa: F401
