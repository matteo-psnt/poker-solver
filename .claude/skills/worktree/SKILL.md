---
name: worktree
description: Set up, run in, merge back and clean up a git worktree of poker-solver — the parallel experiment lines under .claude/worktrees/. Use whenever asked to make a worktree, spin up a wt, run an arm or experiment on a branch, rebase or merge a worktree, or drop one.
---

# Worktrees of poker-solver

Parallel experiment lines. This is the source of truth for the procedure —
`fresh-worktree-setup-gotchas` in memory points here and carries nothing else.
**Correct this file when a step goes void; do not re-record it in memory.**

## Create — base ref first, or you silently lose work

`worktree.baseRef` defaults to `origin/main`, and local `main` runs far ahead of
it. `EnterWorktree({name})` branches from the stale ref and says nothing. It has
bitten at least seven sessions: 11, 25, 29, 68, ~100 and 129 commits dropped —
one worktree had no `console/` in it at all.

Always branch from local `HEAD` explicitly, then enter by path:

    git worktree add -b wt-<name> .claude/worktrees/<name> HEAD
    # then EnterWorktree({path: ".claude/worktrees/<name>"})

Already used `EnterWorktree({name})`? Run `git log --oneline main -3` first
thing. A clean, just-created worktree recovers with `git reset --hard main` —
but only before you have edited anything.

## Then three links, or the tools quietly do the wrong thing

`.claude/` is gitignored, and so is half of each Terraform state, so a fresh
worktree lacks all of it. From inside the worktree, with `P` = primary checkout:

    uv sync --group dev

    # 1. project skills — a worktree has NONE until linked.
    #    Never symlink `.claude` itself: it contains worktrees/ and would recurse.
    mkdir -p .claude && ln -sfn "$P/.claude/skills" .claude/skills

    # 2. BOTH Terraform states. Initialising only infra/ leaves the identical
    #    error, because config.py reads infra/store too.
    ln -sfn "$P/infra/terraform.tfstate"       infra/terraform.tfstate
    ln -sfn "$P/infra/store/terraform.tfstate" infra/store/terraform.tfstate
    (cd infra       && terraform init -input=false)
    (cd infra/store && terraform init -input=false)

Cloud commands (`pool-status`, `submit`, `score`) work from a worktree once
those are in place. Symlinking `.terraform` ITSELF does not work — init must
populate a real directory; symlinking the state FILE is fine. `INFRA_DIR` is a
relative path, which is why this is per-worktree rather than once globally.

Only if you need `npm run gen:types`: `ln -sfn "$P/console/node_modules"
console/node_modules`. `.gitignore` has `console/node_modules/` with a trailing
slash, so it matches a directory and **not** this symlink — it shows up
untracked. Delete the link once types are regenerated.

The `data/combo_abstraction` symlink step is **void**: `data/` is deleted on
purpose and caches resolve to `~/.cache/poker-solver`, shared across worktrees.

## Establish the baseline before editing

    uv run pytest -m "not slow" && uv run pre-commit run --all-files

The only way to tell "my change broke a contract" from "this worktree was never
clean" — it once caught a real ordering bug in `logs.py` that the primary
checkout hides.

## Main moves under you

Parallel sessions commit to `main` mid-flight. Re-check `git log main` **before
merging**, not only at the start; the second check turns a surprise into a
rebase. Take main's version where it solved the same problem — do not re-land a
weaker duplicate. `git cherry` LIES after a restructure; verify by symbol chain,
not filename.

## Merge back

`git push . HEAD:main` is REFUSED from inside a worktree. Do not override
`receive.denyCurrentBranch` — it moves the ref without touching main's working
tree, which then shows the whole changeset as uncommitted deletions.

    # ExitWorktree({action: "keep"}), then from the primary checkout:
    git merge --ff-only wt-<name>

Never `git add -A` — a hook blocks it, because parallel sessions share the
primary checkout and it has swept their work into a commit three times.

## Clean up

A worktree created with `git worktree add` is not session-owned, so
`ExitWorktree({action: "remove"})` refuses it:

    # ExitWorktree({action: "keep"}), then:
    git worktree remove .claude/worktrees/<name>
    git branch -d wt-<name>          # -d, so git refuses if anything is unmerged

Confirm "fully merged" with `git merge-base --is-ancestor`, not `git cherry`.
