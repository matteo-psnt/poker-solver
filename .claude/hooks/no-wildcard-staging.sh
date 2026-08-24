#!/usr/bin/env bash
# Refuse to stage by wildcard in this worktree.
#
# Several Claude sessions share this checkout. `git add -A` (or `.`, or `-u`)
# stages whatever is dirty RIGHT NOW, which includes the other session's
# in-flight edits -- so their half-finished work lands in your commit under your
# message. It has happened three times: `b836d02` absorbed a new module written
# by another session, `46b10aa` absorbed a fix, and one commit here had to be
# unwound with `reset --soft` after sweeping in eight files mid-sweep.
#
# `git commit -a` is the same action with the staging step hidden, so it is
# refused too.
#
# Reads the PreToolUse payload on stdin; prints a deny decision and exits 0 when
# the command sweeps. Anything else prints nothing, which lets the call through.
set -uo pipefail

command=$(jq -r '.tool_input.command // ""' 2>/dev/null) || exit 0
[ -n "$command" ] || exit 0

# `[^;&|]*` keeps each test inside one command segment, so `git status; git add -A`
# is caught while `git add src/a.py && ruff format .` is not: the `.` there
# belongs to a different command.
add_sweep='(^|[;&|(]|[[:space:]])git[[:space:]]+add\b[^;&|]*[[:space:]](-[A-Za-z]*[Aau][A-Za-z]*|--all|--update|\.)([[:space:]]|$)'
commit_sweep='(^|[;&|(]|[[:space:]])git[[:space:]]+commit\b[^;&|]*[[:space:]](-[A-Za-z]*a[A-Za-z]*|--all)([[:space:]]|$)'

if ! printf '%s' "$command" | grep -qE "$add_sweep|$commit_sweep"; then
  exit 0
fi

reason='Wildcard staging is blocked in this worktree.

Parallel Claude sessions share this checkout, so `git add -A` / `.` / `-u` (and
`git commit -a`, which stages the same way) sweep the other session'"'"'s in-flight
edits into your commit. That has happened three times here.

Stage the paths you actually changed:

    git add path/one.py path/two.py

Check what is yours first with `git status --porcelain` and `git diff <path>`,
and re-check immediately before committing -- the tree moves under you.'

jq -n --arg reason "$reason" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: $reason
  }
}'
