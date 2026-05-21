"""What was billed, and what ran.

    billing.py    what Azure actually charged, from Cost Management
    node_time.py  what the pool actually ran, from the task log

Deliberately two sources that are never conflated. The screen once multiplied a
guessed rate by open-ended node intervals and reported $574.61 against an actual
$316.71; billed now comes from the biller and node-time from the record, and
each is reported as itself.
"""
