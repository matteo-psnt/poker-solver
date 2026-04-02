#!/usr/bin/env python3
"""Node-side entry point for one experiment leg. This is what a Batch task runs.

Deliberately thin. It exists to be a *path inside the code tarball* that the
task command line can name, and to put the repo on ``sys.path`` before the
project's environment exists -- nothing else belongs here, because nothing else
here could be tested.

It runs under the NODE's system ``python3``, which on the pinned Ubuntu 22.04
image is 3.10, and BEFORE ``uv sync``. Everything it reaches must therefore be
stdlib-only and 3.10-compatible; see ``src/shared/node/__init__.py``.

Previously ``run_leg.sh``, 677 lines. The publish and fetch rules it carried
now live in :mod:`src.shared.node.archive` with tests, and the argv it built by
hand lives in :mod:`src.shared.node.plan`, checked against the real parsers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.node.runner import main

if __name__ == "__main__":
    sys.exit(main())
