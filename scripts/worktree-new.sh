#!/usr/bin/env bash
# worktree-new.sh — start a feature branch in its own worktree, the first step of the lifecycle merge.sh ends.
# CANON: Platform/templates/worktree-new.sh, copied to <repo>/scripts/worktree-new.sh.
#
# Usage:
#   worktree-new.sh <type> <description> [--offline]
#
#   <type>         a Conventional Commit type: feat, fix, docs, chore, refactor, test, build, ci, perf or revert
#   <description>  kebab-case, for example login-validation
#   --offline      branch from the local main instead of fetching origin/main first
#
# Creates the branch <type>/<description>-YYYY-MM-DD (the date in the local time zone; -2, -3 … when the name
# is taken) in <repository root>/.worktrees/<description>-YYYY-MM-DD from the latest origin/main, makes sure
# /.worktrees/ is excluded in .git/info/exclude (never the tracked .gitignore) and prints the worktree path.
set -euo pipefail

usage() { echo "usage: worktree-new.sh <type> <description> [--offline]" >&2; exit 2; }
TYPES="feat fix docs chore refactor test build ci perf revert"
type="" description="" offline=0
for argument in "$@"; do
  case "$argument" in
    --offline) offline=1 ;;
    -*) usage ;;
    *) if [ -z "$type" ]; then type=$argument; elif [ -z "$description" ]; then description=$argument; else usage; fi ;;
  esac
done
[ -n "$type" ] && [ -n "$description" ] || usage
case " $TYPES " in *" $type "*) ;; *) echo "'$type' is not a Conventional Commit type ($TYPES)" >&2; exit 2 ;; esac
[[ "$description" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || { echo "the description must be kebab-case, e.g. login-validation" >&2; exit 2; }

common=$(git rev-parse --path-format=absolute --git-common-dir)
root=$(dirname "$common")
if [ "$offline" -eq 1 ]; then
  base=main
else
  git -C "$root" fetch --quiet origin main
  base=origin/main
fi
git -C "$root" rev-parse --verify --quiet "$base^{commit}" >/dev/null || { echo "no $base to branch from" >&2; exit 1; }

date=$(date +%F)
name="$description-$date"
suffix=1
while git -C "$root" show-ref --verify --quiet "refs/heads/$type/$name" \
  || git -C "$root" show-ref --verify --quiet "refs/remotes/origin/$type/$name" \
  || [ -e "$root/.worktrees/$name" ]; do
  suffix=$((suffix + 1))
  name="$description-$date-$suffix"
done
branch="$type/$name"
case "$branch" in
  main | master | develop | release | release/*) echo "refusing the protected branch name '$branch'" >&2; exit 2 ;;
esac

mkdir -p "$common/info"
grep -qx '/.worktrees/' "$common/info/exclude" 2>/dev/null || echo '/.worktrees/' >> "$common/info/exclude"
git -C "$root" worktree add --quiet "$root/.worktrees/$name" -b "$branch" "$base"
echo "$root/.worktrees/$name"
