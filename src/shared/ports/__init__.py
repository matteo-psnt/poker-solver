"""Protocols the work depends on, and nothing else.

Nothing in here imports anything from `src`. That is what makes
`pipeline -> ports` legal while `pipeline -> adapters` is forbidden, and it is
asserted by the `ports_declare_they_do_not_do` contract rather than intended.
"""
