"""What an infoset IS, and the two ways a live state is turned into one.

    model.py    InfoSetKey and InfoSet -- the identity and the stored record
    encoder.py  GameState -> InfoSetKey, the string form
    index.py    GameState -> flat row index, the (node_id, bucket) form

Two encodings coexist on purpose. `index.py` is the one the static tree runs
on; `encoder.py` remains because the string key is still what a human reads and
what parity tests compare against. Filed together because a change to what an
infoset is has to be made in all three or in none.
"""
