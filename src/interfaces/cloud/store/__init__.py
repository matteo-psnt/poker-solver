"""The durable share, and the throwaway tree a reader materialises from it.

    share.py      SMB IO against the Azure Files share
    workspace.py  what `--source share` materialises, cached and refcounted

Azure Files has NO atomic rename, so safety here comes from the layout -- one
writer per file -- rather than from the write. That constraint belongs to this
pair and to nothing else in `cloud/`.
"""
