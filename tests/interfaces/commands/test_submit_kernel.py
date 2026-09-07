"""`submit` defaults to pcs, whose iteration is a board -- so a scalar-sized
target with no kernel named is refused rather than queued for two days."""

from __future__ import annotations

import argparse

import pytest

from src.interfaces.commands import submit
from src.interfaces.errors import CommandError


def _args(**over):
    return argparse.Namespace(**({"kernel": None, "to": 4_000} | over))


class TestTheDefaultKernel:
    def test_a_pcs_sized_target_defaults_to_pcs(self):
        assert submit._kernel(_args()) == "pcs"

    def test_a_scalar_sized_target_with_no_kernel_is_refused(self):
        with pytest.raises(CommandError, match="BOARDS"):
            submit._kernel(_args(to=200_000))

    def test_the_ceiling_itself_still_defaults(self):
        assert submit._kernel(_args(to=submit.PCS_DEFAULT_TO_CEILING)) == "pcs"

    def test_an_explicit_kernel_is_never_second_guessed(self):
        assert submit._kernel(_args(kernel="pcs", to=200_000)) == "pcs"
        assert submit._kernel(_args(kernel="scalar", to=200_000)) == "scalar"
