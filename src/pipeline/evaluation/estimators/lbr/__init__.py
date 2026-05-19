"""Local Best Response against the trained HUNL blueprint.

LBR (Lisy & Bowling 2017) plays a concrete, cheap exploiter against the frozen
blueprint, so what it reports is a realizable value and therefore a LOWER BOUND
on true exploitability -- never the exact figure.

The five modules here are one evaluator, split by the axis each varies:
``hunl_local_best_response`` is the driver and owns ``LBRConfig``;
``lookahead_scorer`` and the myopic path are how the exploiter VALUES an action;
``opponent_model`` is what it plays against (blueprint, or blueprint plus the
runtime resolver); ``shadow_state`` is what lets it bet OFF the trained tree at
all; ``lbr_showdown`` settles the hand. Those knobs are not free parameters --
each one defines the comparison tier a result belongs to, and ``ledger.tiers``
derives that tier from the same ``LBRConfig`` the evaluation consumed so two
numbers from different tiers can never be paired.
"""
