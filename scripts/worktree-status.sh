#!/usr/bin/env bash
# worktree-status.sh — every feature worktree under .worktrees/ at a glance, with whether merge.sh can take it.
# CANON: Platform/templates/worktree-status.sh, copied to <repo>/scripts/worktree-status.sh.
#
# Usage:
#   worktree-status.sh [--offline]
#
# Columns: the worktree, its branch, commits ahead of and behind the base (origin/main, or main with --offline
# or without an origin), uncommitted files, and a state: "ready" (clean, with commits to integrate),
# "dirty" (commit or discard first), "empty" (no commits yet) or "detached".
set -euo pipefail

offline=0
case "${1:-}" in
  "") ;;
  --offline) offline=1 ;;
  *) echo "usage: worktree-status.sh [--offline]" >&2; exit 2 ;;
esac
common=$(git rev-parse --path-format=absolute --git-common-dir)
root=$(dirname "$common")
base=main
if [ "$offline" -eq 0 ] && git -C "$root" remote get-url origin >/dev/null 2>&1; then
  git -C "$root" fetch --quiet origin main && base=origin/main
fi

printf '%-40s %-48s %5s %6s %5s  %s\n' WORKTREE BRANCH AHEAD BEHIND DIRTY STATE
found=0
path=""
while IFS= read -r line; do
  case "$line" in
    "worktree "*) path=${line#worktree } ;;
    "") path="" ;;
  esac
  case "$path" in "$root/.worktrees/"*) ;; *) continue ;; esac
  case "$line" in
    "branch "* | detached)
      found=1
      dirty=$(git -C "$path" status --porcelain | wc -l | tr -d ' ')
      if [ "$line" = detached ]; then
        printf '%-40s %-48s %5s %6s %5s  %s\n' "${path#"$root"/}" "(detached)" - - "$dirty" detached
        continue
      fi
      branch=${line#branch refs/heads/}
      read -r behind ahead < <(git -C "$root" rev-list --left-right --count "$base...$branch")
      if [ "$dirty" -gt 0 ]; then state=dirty; elif [ "$ahead" -eq 0 ]; then state=empty; else state="ready for merge.sh"; fi
      printf '%-40s %-48s %5s %6s %5s  %s\n' "${path#"$root"/}" "$branch" "$ahead" "$behind" "$dirty" "$state"
      ;;
  esac
done < <(git -C "$root" worktree list --porcelain; echo)
[ "$found" -eq 1 ] || echo "no worktrees under .worktrees/"
