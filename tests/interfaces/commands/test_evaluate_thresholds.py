import argparse

from src.interfaces.commands.evaluate import add_arguments, threshold_b


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args, _ = parser.parse_known_args(["--run", "r", *argv])
    return args


def test_side_b_is_served_like_side_a_by_default() -> None:
    # Measured 11-17: gates passing only --policy-threshold left the incumbent raw,
    # handing every challenger the +35..+91 mbb/hand purification edge.
    assert threshold_b(_parse("--policy-threshold", "0.35")) == 0.35


def test_an_explicit_side_b_threshold_wins() -> None:
    assert threshold_b(_parse("--policy-threshold", "0.35", "--vs-policy-threshold", "0")) == 0.0
