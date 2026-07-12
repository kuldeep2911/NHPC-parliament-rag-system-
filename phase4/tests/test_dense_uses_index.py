"""
DEPRECATED SHIM — phase4.tests.test_dense_uses_index moved to nhpc_qa.tests.test_dense_uses_index.

The code was relocated into the single nhpc_qa application; the LOGIC is unchanged. This
module re-exports from the new home so existing commands and scripts keep working during
the transition. It will be deleted once the migration is confirmed.

    old:  python -m phase4.tests.test_dense_uses_index
    new:  python -m nhpc_qa.tests.test_dense_uses_index
"""
import warnings as _w
_w.warn("phase4.tests.test_dense_uses_index has moved to nhpc_qa.tests.test_dense_uses_index; use `python -m nhpc_qa.tests.test_dense_uses_index`",
        DeprecationWarning, stacklevel=2)

from nhpc_qa.tests.test_dense_uses_index import *          # noqa: F401,F403
from nhpc_qa.tests.test_dense_uses_index import main       # noqa: F401
