"""Bare-format logging for the library layers (pipeline/engine).

Library code must not ``print()``: it pollutes stdout that machine consumers
(``poker-solver-run --json``) need clean, and forces redirect hacks at the
process edges. Modules log via ``logging.getLogger(__name__)`` instead, and
every process entrypoint (CLI mains, Modal functions, spawn workers) calls
``configure_logging()`` once.

The format is deliberately the bare message — no timestamps, no level/module
prefixes — so terminal output reads exactly like the ``print()`` output it
replaced. Records go to stderr, keeping stdout reserved for payloads.
"""

import logging
import sys


class _DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stderr`` at emit time.

    A plain ``StreamHandler(sys.stderr)`` binds the stream object once, so
    later redirections (pytest ``capsys``, worker-side stream swaps) silently
    write to a dead buffer. Resolving per record follows the current stderr.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass  # always resolve dynamically; ignore the base-class assignment


def configure_logging(level: int = logging.INFO) -> None:
    """Install the bare stderr handler on the ``src`` package logger.

    Idempotent: repeat calls only adjust the level. Handles are attached to
    the package root (not the global root) so third-party loggers keep their
    own configuration, and propagation is cut so nothing double-prints via
    the root logger's handlers.
    """
    package_logger = logging.getLogger("src")
    if not package_logger.handlers:
        handler = _DynamicStderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        package_logger.addHandler(handler)
        package_logger.propagate = False
    package_logger.setLevel(level)
