"""Azure control plane: what to run in the cloud, and how it is dispatched.

Lives under ``interfaces`` on purpose. Nothing in ``pipeline``/``engine``/
``core`` should ever be able to reach Azure, and placing the SDK here makes
that a structural property rather than a convention -- the solver's import
graph cannot pull in a cloud client.

The division of labour this package sits inside:

* **Terraform owns what EXISTS** -- the Batch account, the pool, the share,
  the policy denials. Not this package's business.
* **This package owns what HAPPENS** -- snapshotting the tree, building a task
  spec, creating jobs and tasks, reading their state back.
* **``src/shared/cloudtask/node/`` owns what happens ON the node** -- fetching the
  published checkpoint, guarding and teeing the training process, publishing
  each rung as it appears, and accounting for however the task ended. It sits in
  ``shared`` rather than here because ``pipeline`` reads the task log those
  modules write, and because the node runs the wrapper BEFORE ``uv sync``, so it
  can import neither this package nor anything third-party. ``infra/main.tf``'s start task is the one node-side thing still
  shell: it formats and mounts the data disk before any code exists to run.
"""
