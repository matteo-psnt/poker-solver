#!/usr/bin/env python3
"""Node-side entry point for one experiment task. This is what a Batch task runs.

Deliberately thin. It exists to be a *path inside the code tarball* that the
task command line can name, and to put the repo on ``sys.path`` before the
project's environment exists -- nothing else belongs here, because nothing else
here could be tested.

It runs on the interpreter the pool's start task installs -- not the OS's -- and
BEFORE ``uv sync``. Everything it reaches must therefore be stdlib-only; see
``src/shared/cloudtask/node/__init__.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.cloudtask.node.lifecycle import main

if __name__ == "__main__":
    sys.exit(main())
