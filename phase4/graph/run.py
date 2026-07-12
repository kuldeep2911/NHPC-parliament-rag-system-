"""
DEPRECATED SHIM — phase4.graph.run moved to nhpc_qa.retrieval.graph.run.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase4.graph.run
    new:  nhpc query
"""
import warnings as _w
_w.warn("phase4.graph.run has moved to nhpc_qa.retrieval.graph.run; use `nhpc query`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.retrieval.graph.run import *          # noqa: F401,F403
from nhpc_qa.retrieval.graph.run import main       # noqa: F401
