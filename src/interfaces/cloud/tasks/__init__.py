"""What a Batch task IS, and how one comes to exist.

    spec.py      pure: the task, as a value. The testable core.
    batch.py     the Azure Batch client -- pools, jobs, tasks, their state
    dispatch.py  turning a spec into a queued task

Kept apart from `store/` because submitting work and reading the record are
different jobs with different failure modes, and from `cost/` because what a
task COST is answered by the biller, never by this.
"""
