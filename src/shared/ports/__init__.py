"""Protocols the work depends on, and nothing else.

Nothing in here imports anything from `src`. That is what makes
`pipeline -> ports` legal while `pipeline -> adapters` is forbidden; the
`shared_is_layer_neutral` contract in `.importlinter` asserts it.
"""
