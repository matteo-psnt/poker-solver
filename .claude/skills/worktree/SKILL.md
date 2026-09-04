---
name: worktree
description: Set up, run in, merge back and clean up a git worktree of poker-solver — the parallel experiment lines under .claude/worktrees/. Use whenever asked to make a worktree, spin up a wt, run an arm or experiment on a branch, rebase or merge a worktree, or drop one.
---

# Worktrees of poker-solver

Parallel experiment lines. This is the source of truth for the procedure —
`fresh-worktree-setup-gotchas` in memory points here and carries nothing else.
**Correct this file when a step goes void; do not re-record it in memory.**

## Create

`.claude/settings.json` sets `worktree.baseRef: "head"`, so `EnterWorktree`
branches from local `HEAD` — local `main` runs far ahead of `origin/main`, and
the default base once dropped up to 129 commits without a word. Either works:

    EnterWorktree({name: "<name>"})          # branch worktree-<name>, from HEAD
    git worktree add -b wt-<name> .claude/worktrees/<name> HEAD   # by hand

Whichever you used, `git log --oneline main -1` inside the worktree must show
main's tip. If it does not, `git reset --hard main` — before editing anything.

## Then the links, or the tools quietly do the wrong thing

`.claude/skills/*` is gitignored except this skill, and half of
each Terraform state is too, so a fresh worktree lacks them. From inside the
worktree, with `P` = primary checkout:

    uv sync --group dev

    # 1. the untracked skills, ONE LINK EACH. `.claude/skills` already exists
    #    (worktree/ is tracked), so linking the directory itself
    #    nests a `skills/skills` link and loads nothing — every worktree made
    #    before 09-03 was missing these.
    for s in coding-standards testing tooling; do
      ln -sfn "$P/.claude/skills/$s" ".claude/skills/$s"
    done

    # 2. BOTH Terraform states. Initialising only infra/ leaves the identical
    #    error, because config.py reads infra/store too.
    ln -sfn "$P/infra/terraform.tfstate"       infra/terraform.tfstate
    ln -sfn "$P/infra/store/terraform.tfstate" infra/store/terraform.tfstate
    (cd infra       && terraform init -input=false)
    (cd infra/store && terraform init -input=false)

Once the remote state backend has been migrated (`backend.tf` present in each
root rather than `backend.tf.disabled` -- see "State" in `infra/README.md`),
step 2 is the two `terraform init` lines only: there is no state file to link.

Cloud commands (`pool-status`, `submit`, `score`) work from a worktree once
those are in place. Symlinking `.terraform` ITSELF does not work — init must
populate a real directory; symlinking the state FILE is fine. `INFRA_DIR` is a
relative path, which is why this is per-worktree rather than once globally.

Only if you need `npm run gen:types`: `ln -sfn "$P/console/node_modules"
console/node_modules`. `.gitignore` has `console/node_modules/` with a trailing
slash, so it matches a directory and **not** this symlink — it shows up
untracked. Delete the link once types are regenerated.

## Running commands once you are isolated

`EnterWorktree` isolates the session, and Bash then **refuses any command it
cannot statically prove stays inside the worktree** — "too complex to verify".
That refusal fired 111 times across past sessions, and it costs a whole turn
each. What trips it, all of it ordinary shell:

- `&&` / `;` chains, `for` loops, and `A=x B=y cmd` prefixes
- heredocs — `python3 - <<'PY' … PY` and `git commit -F - <<'MSG'`
- anything writing a path the checker cannot resolve to this worktree

So in an isolated session:

- **One plain command per Bash call.** Split the chain; do not try to smuggle
  it past with a `cd`. Bash's cwd does not persist between calls anyway, so a
  relative path resolves against the PRIMARY checkout, not this one.
- **Edit files with Edit/Write, not `python3 - <<PY`.** The bash-first habit
  is what generates most of these refusals. Pass absolute worktree paths.
- **Commit messages go in a file**, `git commit -F <path>`, not a heredoc.

`CLAUDE.md` is a symlink to `AGENTS.md` in every checkout, and Edit refuses to
write through a symlink. Edit `AGENTS.md`.

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
    git merge --ff-only wt-<name>        # or worktree-<name>

Never `git add -A` — a hook blocks it, because parallel sessions share the
primary checkout and it has swept their work into a commit three times.

## Clean up

A worktree created with `git worktree add` is not session-owned, so
`ExitWorktree({action: "remove"})` refuses it:

    # ExitWorktree({action: "keep"}), then:
    git worktree remove .claude/worktrees/<name>
    git branch -d wt-<name>          # -d, so git refuses if anything is unmerged
    # (an EnterWorktree one: ExitWorktree({action: "remove"}) does both)

Confirm "fully merged" with `git merge-base --is-ancestor`, not `git cherry`.
