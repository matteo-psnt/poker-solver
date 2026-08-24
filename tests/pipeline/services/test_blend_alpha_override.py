"""The blend-alpha override, which the attribution control depends on."""

from __future__ import annotations

from src.interfaces.cli.headless import build_parser
from src.pipeline.services.scoring.matches import _with_resolver_overrides
from tests.test_helpers import make_test_config


class TestAlphaZeroIsNotSwallowed:
    """`alpha=0` makes deployed play exactly the blueprint -- the control that
    attributes an off-tree gap to the resolver ROW rather than to the lookup
    path around it. A falsy-vs-None slip would silently score the default 0.35
    while the arm believed it had turned the resolver off.
    """

    def test_the_flag_reaches_the_config(self):
        args = build_parser().parse_args(
            [
                "evaluate",
                "--run",
                "r",
                "--method",
                "lbr",
                "--opponent",
                "deployed",
                "--resolver-blend-alpha",
                "0",
            ]
        )
        assert args.resolver_blend_alpha == 0.0

        config = make_test_config(seed=42)
        assert config.resolver.policy_blend_alpha != 0.0, "0 has to be a real change"
        overridden = _with_resolver_overrides(
            config,
            leaf_continuation_fraction=None,
            max_iterations=None,
            root_prior_weight=None,
            leaf_rollouts=None,
            blend_alpha=args.resolver_blend_alpha,
        )
        assert overridden.resolver.policy_blend_alpha == 0.0

    def test_omitting_it_keeps_the_run_value(self):
        config = make_test_config(seed=42)
        kept = _with_resolver_overrides(
            config,
            leaf_continuation_fraction=None,
            max_iterations=None,
            root_prior_weight=None,
            leaf_rollouts=None,
            blend_alpha=None,
        )
        assert kept.resolver.policy_blend_alpha == config.resolver.policy_blend_alpha
