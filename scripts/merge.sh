#!/usr/bin/env bash
#
# merge.sh — safely integrate a worktree feature branch, then clean it up.
#
# CANON: this file lives in Platform/templates/merge.sh — do not edit per-repo copies;
# edit here and re-sync to each consumer's scripts/ dir (see templates/README.md).
#
# Usage:
#   merge.sh <branch> <message> [options]
#
# Given a local feature branch, this script operates on the primary working tree:
#   1. validates both branches, worktrees, and remote freshness,
#   2. merges <branch> into the target (default: main) with --no-ff,
#   3. removes the feature worktree,
#   4. atomically deletes the feature branch if its tip did not move.
#
# It refuses before merging when the primary or source worktree is unsafe to touch,
# the target is not a local branch, origin/<target> is ahead or diverged, another
# merge helper is running, or the source is protected (main/master/develop/release/**).
#
# "Unsafe to touch" excludes regenerable build output (build/, node_modules/, .gradle/,
# .kotlin/, … — see REGENERABLE_IGNORED_RE). Those are ignored paths that a rebuild recreates, so
# they are removed with the worktree and reported, never treated as work to preserve.
# Every other ignored file — .env.local, credentials, anything irreplaceable — still
# blocks until you pass --discard-worktree-changes.
#
# Arguments:
#   <branch>          local feature branch to merge and delete (required)
#   <message>         merge commit message, a Conventional Commit such as "feat: add X" (required)
#
# Options:
#   --into <target>                 local target branch (default: main)
#   --discard-worktree-changes      discard uncommitted, untracked, and ignored source files
#                                   after a successful merge
#   --force, -f                     deprecated alias for --discard-worktree-changes
#   --offline                       skip fetch and origin/<target> freshness validation
#   --dry-run                       validate and print the planned actions without changing HEAD,
#                                   local branches, the index, or worktrees (fetch may still run)
#   --                              treat all remaining arguments as positional
#   -h, --help                      show this header
#
# Warnings:
#   When the source carries commits whose changes are already on the target (the same patch id,
#   for example a cherry-pick that reached the target another way), the script lists them and
#   prints the rebase that drops the copies. It still merges and keeps the exit code: picking a
#   hotfix onto the target and merging its branch later is a legitimate flow.
#
# Outcome:
#   A successful run ends with a green "✓" line; every failure ends with a red "✗" line.
#   The glyph is always printed, so captured output stays greppable; only the colour is
#   conditional.
#
# Environment:
#   NO_COLOR        set to any value to print the marks without colour (wins over FORCE_COLOR)
#   FORCE_COLOR     set to any value to colour the marks even when the stream is not a terminal

set -euo pipefail

TARGET="main"
DISCARD_WORKTREE_CHANGES=0
OFFLINE=0
DRY_RUN=0
LEGACY_FORCE=0
BRANCH=""
MESSAGE=""
MSG_SET=0
PARSE_OPTIONS=1
LOCK_HELD=0
LOCK_DIR=""

# Ignored paths that a rebuild recreates. The worktree guard must distinguish these from
# irreplaceable local data: a lint or build run inside the worktree is MANDATORY in this
# workflow, so treating its output as "work to preserve" made the guard fire on every
# single handoff and trained callers to pass --discard-worktree-changes reflexively —
# which is precisely how a real .env.local eventually gets discarded unread.
#
# Matched per path segment against `git status --porcelain --ignored=matching` lines, so
# `backend/build/`, `frontend/node_modules/` and `gradle/x/build/libs/a.jar` all qualify.
# Keep this list conservative: anything absent from it keeps blocking, which is the safe
# direction to be wrong in.
REGENERABLE_IGNORED_RE='^!! (.*/)?((build|dist|target|bin|node_modules|coverage|test-results|playwright-report|__pycache__|\.gradle|\.kotlin|\.next|\.turbo|\.venv|\.pytest_cache)(/.*)?|tsconfig\.tsbuildinfo)$'

# Status lines that must block the merge: everything except regenerable build output.
# Tracked modifications and untracked non-ignored files are never filtered.
blocking_status() {
  [ -n "$1" ] || return 0
  printf '%s\n' "$1" | grep -Ev "$REGENERABLE_IGNORED_RE" || true
}

# Outcome marks. The glyph is always emitted so captured output stays greppable;
# colour is added only for an interactive stream. NO_COLOR suppresses colour entirely
# and wins over FORCE_COLOR, which exists so the coloured path stays testable off a TTY.
#
# The two -t probes must run here, in the main shell: inside a command substitution
# stdout is a pipe, so `[ -t 1 ]` would be false no matter what the caller's stream is.
STDOUT_COLOUR=0
STDERR_COLOUR=0
if [ -z "${NO_COLOR:-}" ]; then
  if [ -n "${FORCE_COLOR:-}" ]; then
    STDOUT_COLOUR=1
    STDERR_COLOUR=1
  else
    if [ -t 1 ]; then STDOUT_COLOUR=1; fi
    if [ -t 2 ]; then STDERR_COLOUR=1; fi
  fi
fi

mark() {
  local glyph="$1" colour="$2" enabled="$3"
  if [ "$enabled" -eq 1 ]; then
    printf '\033[%sm%s\033[0m' "$colour" "$glyph"
  else
    printf '%s' "$glyph"
  fi
}

OK_MARK="$(mark "✓" 32 "$STDOUT_COLOUR")"
FAIL_MARK="$(mark "✗" 31 "$STDERR_COLOUR")"

usage() {
  echo "usage: merge.sh <branch> <message> [--into <target>] [--discard-worktree-changes] [--offline] [--dry-run]" >&2
}

usage_error() {
  echo "$FAIL_MARK $1" >&2
  usage
  exit 2
}

die() {
  echo "$FAIL_MARK $1" >&2
  exit 1
}

cleanup_lock() {
  if [ "$LOCK_HELD" -eq 1 ]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

while [ $# -gt 0 ]; do
  if [ "$PARSE_OPTIONS" -eq 1 ]; then
    case "$1" in
      --discard-worktree-changes)
        DISCARD_WORKTREE_CHANGES=1
        shift
        continue
        ;;
      --force|-f)
        DISCARD_WORKTREE_CHANGES=1
        LEGACY_FORCE=1
        shift
        continue
        ;;
      --offline)
        OFFLINE=1
        shift
        continue
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        continue
        ;;
      --into)
        [ $# -ge 2 ] || usage_error "--into needs a branch name"
        case "$2" in
          -*) usage_error "--into needs a branch name" ;;
        esac
        TARGET="$2"
        shift 2
        continue
        ;;
      --)
        PARSE_OPTIONS=0
        shift
        continue
        ;;
      -h|--help)
        awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"
        exit 0
        ;;
      -*) usage_error "unknown option: $1" ;;
    esac
  fi

  if [ -z "$BRANCH" ]; then
    BRANCH="$1"
  elif [ "$MSG_SET" -eq 0 ]; then
    MESSAGE="$1"
    MSG_SET=1
  else
    usage_error "unexpected extra argument: $1"
  fi
  shift
done

[ -n "$BRANCH" ] || usage_error "feature branch is required"
if [ "$MSG_SET" -ne 1 ] || [[ ! "$MESSAGE" =~ [^[:space:]] ]]; then
  usage_error "merge message is required and must not be blank"
fi
# The merge commit is what release notes and the bump type are drawn from (scripts/release-notes.sh), so it follows
# the same Conventional Commit rule as release.sh.
CONVENTIONAL='^(feat|fix|docs|chore|refactor|test|build|ci|perf|revert)(\([a-z0-9-]+\))?!?: .+'
if [[ ! "$MESSAGE" =~ $CONVENTIONAL ]]; then
  usage_error "merge message must be a Conventional Commit, <type>[(scope)][!]: <description> with a type of feat, fix, docs, chore, refactor, test, build, ci, perf or revert (got \"$MESSAGE\")"
fi
if [ "$LEGACY_FORCE" -eq 1 ]; then
  echo "warning: --force is deprecated; use --discard-worktree-changes" >&2
fi

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 || die "invalid feature branch name '$BRANCH'"
git check-ref-format --branch "$TARGET" >/dev/null 2>&1 || die "invalid target branch name '$TARGET'"

if [ "$BRANCH" = "$TARGET" ]; then
  die "refusing to merge '$BRANCH' into itself"
fi
case "$BRANCH" in
  main|master|develop|release|release/*)
    die "refusing to delete protected branch '$BRANCH'"
    ;;
esac

SOURCE_REF="refs/heads/$BRANCH"
TARGET_REF="refs/heads/$TARGET"
git show-ref --verify --quiet "$SOURCE_REF" || die "feature branch '$BRANCH' does not exist"
git show-ref --verify --quiet "$TARGET_REF" || die "target branch '$TARGET' does not exist"
SOURCE_OID="$(git rev-parse --verify "$SOURCE_REF^{commit}")" || die "feature branch '$BRANCH' does not point to a commit"

COMMON_DIR="$(git rev-parse --git-common-dir)"
case "$COMMON_DIR" in
  /*) ;;
  *) COMMON_DIR="$(cd "$COMMON_DIR" && pwd -P)" ;;
esac
LOCK_DIR="$COMMON_DIR/merge-helper.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [ -d "$LOCK_DIR" ]; then
    die "another merge helper appears to be running (lock: $LOCK_DIR)"
  fi
  die "cannot create merge-helper lock in Git common dir: $LOCK_DIR"
fi
LOCK_HELD=1
trap cleanup_lock EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Git documents the primary working tree as the first porcelain entry.
MAIN_WT="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
[ -n "$MAIN_WT" ] || die "cannot locate the primary working tree"
cd "$MAIN_WT"

BRANCH_WT=""
BRANCH_WT_LOCKED=0
TARGET_WT=""
cur=""
source_matched=0
while IFS= read -r line; do
  case "$line" in
    "worktree "*)
      cur="${line#worktree }"
      ;;
    "branch $SOURCE_REF")
      BRANCH_WT="$cur"
      source_matched=1
      ;;
    "branch $TARGET_REF")
      TARGET_WT="$cur"
      ;;
    locked*)
      if [ "$source_matched" -eq 1 ] && [ "$cur" = "$BRANCH_WT" ]; then
        BRANCH_WT_LOCKED=1
      fi
      ;;
    "")
      source_matched=0
      ;;
  esac
done < <(git worktree list --porcelain)

if [ -n "$TARGET_WT" ] && [ "$TARGET_WT" != "$MAIN_WT" ]; then
  die "target branch '$TARGET' is checked out in another worktree: $TARGET_WT"
fi

MAIN_STATUS="$(git -C "$MAIN_WT" status --porcelain --untracked-files=all)" ||
  die "cannot inspect primary working tree: $MAIN_WT"
if [ -n "$MAIN_STATUS" ]; then
  echo "primary working tree ($MAIN_WT) has uncommitted changes:" >&2
  printf '%s\n' "$MAIN_STATUS" >&2
  die "commit or stash primary working tree changes first"
fi

SOURCE_STATUS=""
if [ -n "$BRANCH_WT" ] && [ "$BRANCH_WT" != "$MAIN_WT" ]; then
  if [ "$BRANCH_WT_LOCKED" -eq 1 ]; then
    die "source worktree is locked and cannot be cleaned up: $BRANCH_WT"
  fi
  SOURCE_STATUS_ALL="$(git -C "$BRANCH_WT" status --porcelain --untracked-files=all --ignored=matching)" ||
    die "cannot inspect source worktree: $BRANCH_WT"
  SOURCE_STATUS="$(blocking_status "$SOURCE_STATUS_ALL")"
  if [ -n "$SOURCE_STATUS" ]; then
    if [ "$DISCARD_WORKTREE_CHANGES" -ne 1 ]; then
      echo "source worktree ($BRANCH_WT) has local or ignored files:" >&2
      printf '%s\n' "$SOURCE_STATUS" >&2
      die "commit/stash them, or explicitly use --discard-worktree-changes"
    fi
    echo "warning: source worktree local and ignored files will be discarded after a successful merge:" >&2
    printf '%s\n' "$SOURCE_STATUS" >&2
  elif [ -n "$SOURCE_STATUS_ALL" ]; then
    echo ">> source worktree holds only regenerable build output; it goes with the worktree:" >&2
    printf '%s\n' "$SOURCE_STATUS_ALL" >&2
  fi
fi

if [ "$OFFLINE" -eq 1 ]; then
  echo ">> offline mode: skipping origin/$TARGET freshness check"
else
  git remote get-url origin >/dev/null 2>&1 ||
    die "remote 'origin' is not configured; configure it or explicitly use --offline"
  echo ">> fetching origin/$TARGET"
  if ! git fetch --quiet origin "+refs/heads/$TARGET:refs/remotes/origin/$TARGET"; then
    die "cannot fetch origin/$TARGET; retry when the remote is available or explicitly use --offline"
  fi
  REMOTE_TARGET_REF="refs/remotes/origin/$TARGET"
  git show-ref --verify --quiet "$REMOTE_TARGET_REF" || die "remote branch origin/$TARGET does not exist"
  if ! git merge-base --is-ancestor "$REMOTE_TARGET_REF" "$TARGET_REF"; then
    die "local target '$TARGET' is behind or diverged from origin/$TARGET; synchronize it before merging"
  fi
fi

# Commity źródła o patch-id już obecnym na celu (`git cherry` → „-”): po merge'u historia ma obie kopie.
DUPLICATES="$(git cherry "$TARGET_REF" "$SOURCE_REF" | sed -n 's/^- //p')"
if [ -n "$DUPLICATES" ]; then
  echo "!! warning: $BRANCH re-applies $(grep -c . <<<"$DUPLICATES") commit(s) whose changes are already on $TARGET:" >&2
  while IFS= read -r oid; do git log -1 --format='     %h %s' "$oid" >&2; done <<<"$DUPLICATES"
  if [ -n "$BRANCH_WT" ]; then
    printf '   both copies stay in the history; to drop them first: git -C %q rebase %q\n' "$BRANCH_WT" "$TARGET" >&2
  else
    printf '   both copies stay in the history; to drop them first: git rebase %q %q\n' "$TARGET" "$BRANCH" >&2
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  TARGET_OID="$(git rev-parse --verify "$TARGET_REF^{commit}")"
  COMMIT_COUNT="$(git rev-list --count "$TARGET_REF..$SOURCE_REF")"
  echo ">> DRY RUN — no local branch, HEAD, index, or worktree changes will be made"
  echo "   source:   $BRANCH ($SOURCE_OID)"
  echo "   target:   $TARGET ($TARGET_OID)"
  echo "   primary:  $MAIN_WT"
  echo "   worktree: ${BRANCH_WT:-<none>}"
  echo "   commits:  $COMMIT_COUNT source commit(s) not in target"
  if [ "$DISCARD_WORKTREE_CHANGES" -eq 1 ]; then
    echo "   cleanup:  remove source worktree with explicit discard, then delete unchanged source ref"
  else
    echo "   cleanup:  remove clean source worktree, then delete unchanged source ref"
  fi
  exit 0
fi

echo ">> switching $MAIN_WT to $TARGET"
git -C "$MAIN_WT" switch --no-guess "$TARGET"
ACTUAL_TARGET="$(git -C "$MAIN_WT" symbolic-ref --quiet --short HEAD)" ||
  die "primary working tree unexpectedly entered detached HEAD"
[ "$ACTUAL_TARGET" = "$TARGET" ] || die "expected target '$TARGET', but primary working tree is on '$ACTUAL_TARGET'"

echo ">> merging $BRANCH into $TARGET (--no-ff)"
merge_rc=0
git -C "$MAIN_WT" merge --no-ff -m "$MESSAGE" "$SOURCE_REF" || merge_rc=$?
if [ "$merge_rc" -ne 0 ]; then
  UNMERGED_PATHS="$(git -C "$MAIN_WT" diff --name-only --diff-filter=U)"
  if [ -n "$UNMERGED_PATHS" ]; then
    echo "$FAIL_MARK merge conflict — resolve the unmerged paths and commit, or abort:" >&2
    printf '     git -C %q status\n' "$MAIN_WT" >&2
    printf '     git -C %q merge --abort\n' "$MAIN_WT" >&2
  else
    echo "$FAIL_MARK merge failed with exit code $merge_rc; source branch and worktree were preserved" >&2
    if git -C "$MAIN_WT" rev-parse --verify --quiet MERGE_HEAD >/dev/null; then
      echo "   Git left an in-progress merge; inspect it or abort with:" >&2
      printf '     git -C %q merge --abort\n' "$MAIN_WT" >&2
    fi
  fi
  exit "$merge_rc"
fi

if [ -n "$BRANCH_WT" ] && [ "$BRANCH_WT" != "$MAIN_WT" ]; then
  echo ">> removing source worktree $BRANCH_WT"
  if [ "$DISCARD_WORKTREE_CHANGES" -eq 1 ]; then
    if ! git worktree remove --force "$BRANCH_WT"; then
      die "merge succeeded, but source worktree cleanup failed; branch '$BRANCH' was preserved"
    fi
  else
    CURRENT_SOURCE_STATUS_ALL="$(git -C "$BRANCH_WT" status --porcelain --untracked-files=all --ignored=matching)" ||
      die "merge succeeded, but source worktree can no longer be inspected; branch '$BRANCH' was preserved"
    CURRENT_SOURCE_STATUS="$(blocking_status "$CURRENT_SOURCE_STATUS_ALL")"
    if [ -n "$CURRENT_SOURCE_STATUS" ]; then
      echo "merge succeeded, but source worktree changed during the merge; cleanup was skipped:" >&2
      printf '%s\n' "$CURRENT_SOURCE_STATUS" >&2
      die "branch '$BRANCH' and source worktree were preserved"
    fi
    # Build output never blocks the merge, but `git worktree remove` still refuses to
    # delete a tree that holds it, so the removal itself has to be forced.
    if [ -n "$CURRENT_SOURCE_STATUS_ALL" ]; then
      if ! git worktree remove --force "$BRANCH_WT"; then
        die "merge succeeded, but source worktree cleanup failed; branch '$BRANCH' was preserved"
      fi
    elif ! git worktree remove "$BRANCH_WT"; then
      die "merge succeeded, but source worktree cleanup failed; branch '$BRANCH' was preserved"
    fi
  fi
fi

if ! git merge-base --is-ancestor "$SOURCE_OID" "$TARGET_REF"; then
  die "refusing to delete '$BRANCH': its original tip is not merged into '$TARGET'"
fi
CURRENT_SOURCE_OID="$(git rev-parse --verify "$SOURCE_REF^{commit}")" ||
  die "source ref '$BRANCH' disappeared during cleanup"
if [ "$CURRENT_SOURCE_OID" != "$SOURCE_OID" ]; then
  die "refusing to delete '$BRANCH': its tip moved during the merge"
fi

echo ">> deleting unchanged source branch $BRANCH"
git update-ref -d "$SOURCE_REF" "$SOURCE_OID" ||
  die "refusing to delete '$BRANCH': its ref changed concurrently"

branch_config_rc=0
branch_config_output="$(LC_ALL=C git config --remove-section "branch.$BRANCH" 2>&1)" || branch_config_rc=$?
if [ "$branch_config_rc" -ne 0 ]; then
  case "$branch_config_output" in
    *"no such section"*) ;;
    *) die "source branch was deleted, but its Git config could not be removed: $branch_config_output" ;;
  esac
fi

echo "$OK_MARK merged $BRANCH into $TARGET and cleaned up"
