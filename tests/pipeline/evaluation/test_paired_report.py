"""Tests for the shared paired-LBR comparison report."""

import pytest

from src.pipeline.evaluation.paired_report import report_paired_lbr
from src.shared.log import configure_logging


@pytest.fixture(autouse=True)
def _bare_logging():
    # The report goes through the library logger; install the bare stderr handler
    # (idempotent, resolves sys.stderr dynamically) so capsys sees it in .err.
    configure_logging()


def _results(base_seed: int, samples: list[float]) -> dict:
    return {
        "base_seed": base_seed,
        "pair_samples_mbb": samples,
        "exploitability_mbb": sum(samples) / len(samples),
        "std_error_mbb": 1.0,
    }


def test_refuses_unpaired_evals():
    with pytest.raises(RuntimeError, match="not paired"):
        report_paired_lbr(
            ("a", _results(1, [10.0, 20.0])),
            ("b", _results(2, [10.0, 20.0])),
            diff_label="A - B",
            better_labels=("b", "a"),
        )


def test_reports_comparison_and_verdict(capsys):
    # Arm B beats arm A on every deal by a wide, consistent margin, so the
    # paired difference is positive and significant: verdict names label[0].
    comparison = report_paired_lbr(
        ("arm-a", _results(7, [110.0, 121.0, 130.0, 139.0])),
        ("arm-b", _results(7, [10.0, 22.0, 30.0, 41.0])),
        diff_label="A - B",
        better_labels=("arm-b", "arm-a"),
        show_pairing_gain=True,
    )

    assert comparison["mean_diff"] == pytest.approx(99.25)
    assert comparison["is_significant"]

    out = capsys.readouterr().err
    assert "PAIRED DIFFERENCE (A - B, 4 common deals)" in out
    assert "pairing gain" in out
    assert "VERDICT: arm-b is significantly less exploitable" in out


def test_insignificant_difference_has_no_winner(capsys):
    comparison = report_paired_lbr(
        ("arm-a", _results(7, [10.0, -12.0, 11.0, -9.0])),
        ("arm-b", _results(7, [-11.0, 10.0, -10.0, 12.0])),
        diff_label="A - B",
        better_labels=("arm-b", "arm-a"),
    )

    assert not comparison["is_significant"]
    out = capsys.readouterr().err
    assert "no significant difference" in out
    assert "pairing gain" not in out
