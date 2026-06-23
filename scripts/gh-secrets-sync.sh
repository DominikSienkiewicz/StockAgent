#!/usr/bin/env bash
#
# gh-secrets-sync.sh — wypycha sekrety/zmienne GitHub Actions z lokalnego pliku .env.
#
# Wypychane są tylko klucze faktycznie odwoływane przez workflowy: klucz odwoływany jako
# secrets.* staje się `gh secret set`, jako vars.* — `gh variable set`. Pozostałe klucze
# z .env są ignorowane. Lista wymaganych jest wyprowadzana na żywo z
# .github/workflows/*.yml, więc trzyma się w synchronie z workflowami.
#
# Uwaga StockAgent: workflowy odwołują się WYŁĄCZNIE do secrets.* (config niewrażliwy żyje
# w config.toml), więc realnie skrypt wypycha tylko sekrety; lista variables jest pusta.
# Klucze z .env, które należą do config.toml (np. TELEGRAM_*, FRED_API_KEY gdy nieużyte
# w CI), trafiają do "ignored" i nigdy nie są wypychane.
#
# Wartości są przekazywane do gh przez stdin, nigdy jako argumenty (więc nie pojawiają się
# w `ps`/historii powłoki). Domyślnie plan jest drukowany i potwierdzany przed zastosowaniem.
#
# Usage:   scripts/gh-secrets-sync.sh [-f .env] [-R owner/repo] [--dry-run] [-y]
#   -f, --file     plik env do odczytu (default: <repo>/.env)
#   -R, --repo     docelowe repo owner/name (default: gh auto-detect z git remote)
#       --dry-run  pokaż co zostałoby ustawione, niczego nie zmieniaj
#   -y, --yes      zastosuj bez pytania o potwierdzenie
#
# Exit: 0 = zastosowane/czyste · 1 = jedno lub więcej wywołań set zawiodło · 2 = błąd konfiguracji
#
set -euo pipefail

ENV_FILE=""
REPO=""
DRY_RUN=0
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file) ENV_FILE="${2:-}"; shift 2 ;;
    -R|--repo) REPO="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF_DIR="$ROOT/.github/workflows"
[[ -n "$ENV_FILE" ]] || ENV_FILE="$ROOT/.env"

if [[ -t 1 ]]; then
  GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  GREEN=""; RED=""; YELLOW=""; BOLD=""; DIM=""; RESET=""
fi

die() { echo "${RED}error:${RESET} $*" >&2; exit 2; }

# Odczyt wartości pojedynczego klucza z pliku env (ostatnie przypisanie wygrywa). Zwraca surową
# wartość po pierwszym '=', minus jedna warstwa otaczających cudzysłowów i końcowy CR. Komentarze
# inline NIE są usuwane (wartości brane dosłownie), by nie uszkodzić URL-i.
get_env_value() {
  local key="$1" file="$2" line val
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$file" | tail -n1 || true)"
  [[ -z "$line" ]] && return 1
  line="${line#"${line%%[![:space:]]*}"}"   # usuń wiodące białe znaki
  line="${line#export }"
  val="${line#*=}"
  val="${val%$'\r'}"                          # usuń końcowy CR (pliki CRLF)
  case "$val" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
    *) ;;  # bez otaczających cudzysłowów — wartość bierzemy dosłownie
  esac
  printf '%s' "$val"
}

command -v gh >/dev/null 2>&1 || die "GitHub CLI (gh) not found — install from https://cli.github.com"
[[ -d "$WF_DIR" ]] || die "no workflows directory at $WF_DIR"
[[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE (copy .env.example to .env first)"

# Preferuj GH_TOKEN z pliku env do każdego wywołania gh (nadpisuje zbyt wąski token już
# wyeksportowany w powłoce). Eksport tylko do tego procesu + jego dzieci, nigdy do Twojej powłoki.
_ENV_GH_TOKEN="$(get_env_value GH_TOKEN "$ENV_FILE" 2>/dev/null || true)"
[[ -n "${_ENV_GH_TOKEN:-}" ]] && export GH_TOKEN="$_ENV_GH_TOKEN"

gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login (or set GH_TOKEN in $ENV_FILE)"

REPO_FLAG=()
if [[ -n "$REPO" ]]; then
  REPO_FLAG=(--repo "$REPO")
  REPO_LABEL="$REPO"
else
  REPO_LABEL="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo '(from git remote)')"
fi

# `|| true`: gdy żaden workflow nie odwołuje się do danego prefiksu (StockAgent nie używa
# vars.* — config niewrażliwy żyje w config.toml), grep zwraca 1 i pod `set -o pipefail`
# wywaliłby skrypt. Pusta lista to poprawny stan, nie błąd.
discover() {
  local prefix="$1"
  grep -rhoE "$prefix\.[A-Za-z0-9_]+" "$WF_DIR" 2>/dev/null \
    | sed "s/^$prefix\.//" | grep -vx 'GITHUB_TOKEN' | sort -u || true
}

# Zbuduj plan w równoległych indeksowanych tablicach (bash 3.2 nie ma tablic asocjacyjnych).
PLAN_NOUN=(); PLAN_NAME=(); PLAN_VALUE=(); PLAN_MASK=()
N=0
SKIPPED=""

stage_kind() {
  # $1 = rzeczownik gh (secret|variable) · $2 = prefiks workflow (secrets|vars) · $3 = mask(1/0)
  local noun="$1" ref="$2" mask="$3" required name value
  required="$(discover "$ref")"
  [[ -z "$required" ]] && return 0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if value="$(get_env_value "$name" "$ENV_FILE")" && [[ -n "$value" ]]; then
      PLAN_NOUN[N]="$noun"; PLAN_NAME[N]="$name"; PLAN_VALUE[N]="$value"; PLAN_MASK[N]="$mask"
      N=$((N + 1))
    else
      SKIPPED="${SKIPPED}  ${YELLOW}skip${RESET} ${name} ${DIM}(absent or empty in $(basename "$ENV_FILE"))${RESET}
"
    fi
  done <<EOF
$required
EOF
}

stage_kind secret   secrets 1
stage_kind variable vars    0

# Klucze z .env nieodwoływane przez żaden workflow — raportowane, nigdy nie wypychane.
required_all="$(printf '%s\n%s\n' "$(discover secrets)" "$(discover vars)" | sort -u)"
env_keys="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$ENV_FILE" \
  | sed -E 's/^[[:space:]]*(export[[:space:]]+)?//; s/=.*//; s/[[:space:]]//g' | sort -u)"
# GH_TOKEN/GITHUB_TOKEN to własny auth skryptu do gh, nigdy kandydaci do wypchnięcia — usuń z "ignored".
ignored="$(comm -23 <(printf '%s\n' "$env_keys") <(printf '%s\n' "$required_all") \
  | grep -v '^$' | grep -vxE 'GH_TOKEN|GITHUB_TOKEN' || true)"

echo "${BOLD}StockAgent — push secrets/variables from $(basename "$ENV_FILE")${RESET}"
echo "Repo: ${REPO_LABEL}   Source: ${ENV_FILE}"
[[ "$DRY_RUN" = "1" ]] && echo "${YELLOW}(dry-run — nothing will be changed)${RESET}"
echo
echo "${BOLD}Plan${RESET}"
if [[ "$N" -eq 0 ]]; then
  echo "  ${DIM}— nothing to push —${RESET}"
else
  i=0
  while [[ "$i" -lt "$N" ]]; do
    if [[ "${PLAN_MASK[i]}" = "1" ]]; then
      shown="•••• ${DIM}(${#PLAN_VALUE[i]} chars)${RESET}"
    else
      shown="${PLAN_VALUE[i]}"
    fi
    echo "  ${GREEN}set${RESET} ${PLAN_NOUN[i]} ${BOLD}${PLAN_NAME[i]}${RESET} = ${shown}"
    i=$((i + 1))
  done
fi
[[ -n "$SKIPPED" ]] && printf '%s' "$SKIPPED"
[[ -n "$ignored" ]] && echo "  ${DIM}ignored (not used by any workflow): $(printf '%s' "$ignored" | tr '\n' ' ')${RESET}"
echo

if [[ "$DRY_RUN" = "1" ]] || [[ "$N" -eq 0 ]]; then
  exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
  printf 'Apply %s change(s) to %s? [y/N] ' "$N" "$REPO_LABEL"
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

FAIL=0
i=0
while [[ "$i" -lt "$N" ]]; do
  if err="$(printf '%s' "${PLAN_VALUE[i]}" | gh "${PLAN_NOUN[i]}" set "${PLAN_NAME[i]}" "${REPO_FLAG[@]+"${REPO_FLAG[@]}"}" 2>&1)"; then
    echo "  ${GREEN}✓${RESET} ${PLAN_NOUN[i]} ${PLAN_NAME[i]}"
  else
    echo "  ${RED}✗ FAILED${RESET} ${PLAN_NOUN[i]} ${PLAN_NAME[i]} ${DIM}— ${err}${RESET}"
    FAIL=$((FAIL + 1))
  fi
  i=$((i + 1))
done
echo
if [[ "$FAIL" -gt 0 ]]; then
  echo "${RED}${FAIL} of ${N} failed.${RESET}"
  if [[ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]]; then
    echo "${YELLOW}GH_TOKEN/GITHUB_TOKEN is set${RESET} and overrides your 'gh auth login' token —"
    echo "  ${DIM}a fine-grained PAT needs 'Secrets: Read and write' (and 'Variables: R/W') to set these.${RESET}"
    echo "  ${DIM}Retry with your keyring token:  GH_TOKEN= GITHUB_TOKEN= ${0##*/}${RESET}"
  fi
  exit 1
fi
echo "${GREEN}Done — ${N} item(s) set on ${REPO_LABEL}.${RESET}"
exit 0
