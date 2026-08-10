"""Reading a strategy out of a trained blueprint.

    lookup.py   the single definition of "the strategy at this state"
    source.py   the seam a consumer asks through, `(state, bucket) -> InfoSet`

Both exist to stop consumers re-deriving the lookup themselves: the restrict
-then-normalize sequence was once reimplemented at six call sites, each with its
own choice of which restriction to apply.
"""
