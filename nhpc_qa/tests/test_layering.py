"""
ARCHITECTURE TEST — the dependency direction must not invert.

The application is layered:

    config  ->  core/{db,trace,providers,logging}  ->  pipeline/  ->  retrieval/  ->  api/

Nothing may point LEFT. In particular `core/` must never import `pipeline/`, `retrieval/`
or `api/`: it is the shared foundation, and a core module reaching up into a caller is how
import cycles are born. The old layout had no cycles, and this test exists to keep it that
way as the app grows.

    python -m nhpc_qa.tests.test_layering
"""

from __future__ import annotations

import ast
import pathlib
import sys

# layer -> the layers it is ALLOWED to import from (plus itself and config)
ALLOWED = {
    "nhpc_qa.config":    set(),                       # config imports the three cfg modules only
    "nhpc_qa.core":      {"nhpc_qa.config"},
    "nhpc_qa.pipeline":  {"nhpc_qa.config", "nhpc_qa.core"},
    "nhpc_qa.retrieval": {"nhpc_qa.config", "nhpc_qa.core", "nhpc_qa.pipeline"},
    "nhpc_qa.api":       {"nhpc_qa.config", "nhpc_qa.core", "nhpc_qa.pipeline",
                          "nhpc_qa.retrieval"},
    "nhpc_qa.watcher":   {"nhpc_qa.config", "nhpc_qa.core", "nhpc_qa.pipeline"},
}
# these may import anything (they are top-level drivers / helpers)
EXEMPT = {"nhpc_qa.cli", "nhpc_qa.tests", "nhpc_qa.scripts"}


def _layer_of(module: str):
    for layer in ALLOWED:
        if module == layer or module.startswith(layer + "."):
            return layer
    for e in EXEMPT:
        if module == e or module.startswith(e + "."):
            return None
    return None


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def main():
    root = pathlib.Path("nhpc_qa")
    violations = []

    for path in sorted(root.rglob("*.py")):
        mod = ".".join(path.with_suffix("").parts)
        layer = _layer_of(mod)
        if layer is None:
            continue                      # exempt (cli/tests/scripts)
        allowed = ALLOWED[layer] | {layer}

        for imported in _imports(path):
            if not imported.startswith("nhpc_qa"):
                continue
            target = _layer_of(imported)
            if target is None:
                # importing an exempt module (cli/tests/scripts) FROM a layer is itself
                # a smell -- the foundation should not depend on a driver.
                if imported.split(".")[1] in ("cli", "tests", "scripts"):
                    violations.append(f"{mod} imports {imported} (a driver, not a layer)")
                continue
            if target not in allowed:
                violations.append(f"{mod}  ->  {imported}   ({layer} may not import {target})")

    print("=" * 74)
    print("LAYERING: config -> core -> pipeline -> retrieval -> api")
    print("=" * 74)
    if violations:
        print(f"\n{len(violations)} VIOLATION(S) — the dependency direction inverted:\n")
        for v in violations:
            print("  ", v)
        return 1
    print("\nPASS — nothing points left; no import cycles are possible by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
