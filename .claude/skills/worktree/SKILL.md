---
name: worktree
description: Set up, run in, merge back and clean up a git worktree of poker-solver — the parallel experiment lines under .claude/worktrees/. Use whenever asked to make a worktree, spin up a wt, run an arm or experiment on a branch, rebase or merge a worktree, or drop one.
---

# Worktrees of poker-solver

Parallel experiment lines. This is the source of truth for the procedure —
`fresh-worktree-setup-gotchas` in memory points here and carries nothing else.
**Correct this file when a step goes void; do not re-record it in memory.**

## Create

From the primary checkout:

    just wt <name>        # .claude/worktrees/<name>, branch wt-<name>, from HEAD

One recipe, because the by-hand list it replaced was six steps and a skipped
one costs a whole gate run: the console hook died on `openapi-typescript:
command not found` on 12 session-days and Terraform on "required plugins" on 8.
What it does, and why each step is there:

- `git worktree add … HEAD` — local `main` runs far ahead of `origin/main`,
  and the default base once dropped 129 commits without a word.
- `uv sync --group dev`.
- one symlink PER SKILL under `.claude/skills/` (everything but this one).
  `.claude/skills` already exists because `worktree/` is tracked, so linking
  the directory itself nests a `skills/skills` link that loads nothing.
- `terraform init` on ALL THREE roots: `config.py` reads `infra/` and
  `infra/store`, and anything touching the blueprint box reads `infra/serve`.
  State is remote, so init only fetches providers; `INFRA_DIR` is relative,
  which is why this is per-worktree and `.terraform` cannot be symlinked.
- `console/node_modules -> primary` — the pre-commit console hook runs
  `npm run ci` and exits 127 without it. `.gitignore` ignores the link.

It refuses to run from inside a worktree, or over a name that exists.

The alternative is `EnterWorktree({name})` (`.claude/settings.json` sets
`worktree.baseRef: "head"`, so it branches from local HEAD too), followed by
the same setup by hand; prefer the recipe. Either way, `git log --oneline
main -1` inside the worktree must show main's tip — if not, `git reset --hard
main` before editing anything.

`CLAUDE.md` is a symlink to `AGENTS.md` in every checkout, and Edit refuses to
write through a symlink. Edit `AGENTS.md`.

## Running commands once you are isolated

`EnterWorktree` isolates the session, and Bash then **refuses any command it
cannot statically prove stays inside the worktree** — "too complex to verify".
That refusal fired 101 times across 56 sessions, a whole turn each. What trips
it, all of it ordinary shell:

- `&&` / `;` chains, `for` loops, and `A=x B=y cmd` prefixes
- heredocs — `python3 - <<'PY' … PY` and `git commit -F - <<'MSG'`
- any write to a path outside the worktree: the scratchpad, `/tmp`, the
  memory directory, the primary checkout

So in an isolated session:

- **One plain command per Bash call.** Bash's cwd does not persist between
  calls, so a relative path resolves against the PRIMARY checkout, not this one:
  pass absolute worktree paths.
- **Edit files with Edit/Write, not `python3 - <<PY`.** Memory files are
  written with the Write tool too; scratch goes under the worktree's `scratch/`.
- **Commit messages go in a file**, `git commit -F <path>`, not a heredoc.

A worktree made with `just wt` is not session-isolated: ordinary shell works,
and absolute paths are still the habit that keeps a `cd` from landing in the
primary checkout.

## Establish the baseline before editing

    uv run pytest -m "not slow" && uv run pre-commit run --all-files

The only way to tell "my change broke a contract" from "this worktree was never
clean" — it once caught a real ordering bug in `logs.py` that the primary
checkout hides. The fast gate is 83 s median and 156 s p90 over 485 measured
runs: give it a 300 s timeout, and pre-commit 600 s.

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

    # ExitWorktree({action: "keep"}) if the session was isolated, then from
    # the primary checkout:
    git merge --squash wt-<name>         # or worktree-<name>
    git commit                           # one commit, carrying the result

A branch is dozens of agent commits and `main` carries one per change, so the
merge is a SQUASH. The branch is then not an ancestor of `main`: `git branch -d`
refuses it and `git merge-base --is-ancestor` reads "unmerged", and `git diff
main wt-<name>` goes stale the moment main moves again. The proof that
survives both: merging the branch would change nothing, which `just wt-rm`
checks with `git merge-tree`.

Never `git add -A` — a hook blocks it, because parallel sessions share the
primary checkout and it has swept their work into a commit three times.

## Clean up

From the primary checkout:

    just wt-rm <name>

It refuses while the worktree has uncommitted changes or the branch still
holds something main does not (a `git merge-tree` of the two is not main's
tree), then removes the worktree and `git branch -D`s the branch. A worktree made by `EnterWorktree` can instead go through
`ExitWorktree({action: "remove"})`, which does both.
