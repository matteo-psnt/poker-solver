"""Azure control plane: what to run in the cloud, and how it is dispatched.

Lives under ``interfaces`` on purpose. Nothing in ``pipeline``/``engine``/
``core`` should ever be able to reach Azure, and placing the SDK here makes
that a structural property rather than a convention -- the solver's import
graph cannot pull in a cloud client.

The division of labour this package sits inside:

* **Terraform owns what EXISTS** -- the Batch account, the pool, the share,
  the policy denials. Not this package's business.
* **This package owns what HAPPENS** -- snapshotting the tree, building a leg
  spec, creating jobs and tasks, reading their state back.
* **``infra/run_leg.sh`` owns what happens ON the node** -- disk discovery,
  mount handling, publish-on-exit traps. It stays shell because it wraps the
  Python process rather than living inside it.
"""
