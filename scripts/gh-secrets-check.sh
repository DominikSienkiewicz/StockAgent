#!/usr/bin/env bash
#
# gh-secrets-check.sh — raportuje, które sekrety/zmienne GitHub Actions są potrzebne
# temu projektowi i które są faktycznie ustawione na repo.
#
# Lista "potrzebnych" jest wyprowadzana na żywo z .github/workflows/*.yml (każde
# odwołanie secrets.* i vars.*), więc nigdy nie odjeżdża od workflowów. GITHUB_TOKEN
# jest wykluczony (GitHub wstrzykuje go automatycznie).
#
# Uwaga StockAgent: workflowy wstrzykują WYŁĄCZNIE sekrety — cały config niewrażliwy
# żyje w commitowanym config.toml, więc sekcja "Variables" zwykle jest pusta. To OK.
#
# Read-only: woła tylko `gh secret list` / `gh variable list`. Niczego nie ustawia.
#
# Auth: jeśli plik env (domyślnie <repo>/.env) definiuje GH_TOKEN, skrypt używa go do
# każdego wywołania gh — działa więc niezależnie od auth gh w Twojej powłoce i nadpisuje
# zbyt wąski GH_TOKEN/GITHUB_TOKEN wyeksportowany globalnie. Pusty/brak → fallback do `gh auth login`.
#
# Usage:   scripts/gh-secrets-check.sh [-R owner/repo] [-f .env]
# Exit:    0 = wszystko wymagane obecne · 1 = czegoś brakuje · 2 = błąd konfiguracji
#
set -euo pipefail

REPO=""
ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -R|--repo) REPO="${2:-}"; shift 2 ;;
    -f|--file) ENV_FILE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF_DIR="$ROOT/.github/workflows"
[ -n "$ENV_FILE" ] || ENV_FILE="$ROOT/.env"

if [ -t 1 ]; then
  GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  GREEN=""; RED=""; YELLOW=""; BOLD=""; DIM=""; RESET=""
fi

die() { echo "${RED}error:${RESET} $*" >&2; exit 2; }

# Odczyt wartości pojedynczego klucza z pliku env (ostatnie przypisanie wygrywa): wszystko po
# pierwszym '=', minus jedna warstwa otaczających cudzysłowów i końcowy CR. Komentarze inline zostają.
get_env_value() {
  local key="$1" file="$2" line val
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$file" | tail -n1 || true)"
  [ -z "$line" ] && return 1
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line#export }"
  val="${line#*=}"
  val="${val%$'\r'}"
  case "$val" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
  esac
  printf '%s' "$val"
}

# Preferuj GH_TOKEN z pliku env do każdego wywołania gh (nadpisuje zbyt wąski token już
# wyeksportowany w powłoce). Eksport tylko do tego procesu + jego dzieci, nigdy do Twojej powłoki.
if [ -f "$ENV_FILE" ]; then
  _ENV_GH_TOKEN="$(get_env_value GH_TOKEN "$ENV_FILE" 2>/dev/null || true)"
  [ -n "${_ENV_GH_TOKEN:-}" ] && export GH_TOKEN="$_ENV_GH_TOKEN"
fi

command -v gh >/dev/null 2>&1 || die "GitHub CLI (gh) not found — install from https://cli.github.com"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login (or set GH_TOKEN in $ENV_FILE)"
[ -d "$WF_DIR" ] || die "no workflows directory at $WF_DIR"

REPO_FLAG=()
if [ -n "$REPO" ]; then
  REPO_FLAG=(--repo "$REPO")
  REPO_LABEL="$REPO"
else
  REPO_LABEL="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo '(from git remote)')"
fi

# Nazwy odwoływane jako <prefix>.NAME we wszystkich plikach workflow ("secrets" lub "vars").
# `|| true`: gdy żaden workflow nie odwołuje się do danego prefiksu (StockAgent nie używa
# vars.* — config niewrażliwy żyje w config.toml), grep zwraca 1 i pod `set -o pipefail`
# wywaliłby skrypt. Pusta lista to poprawny stan, nie błąd.
discover() {
  grep -rhoE "$1\.[A-Za-z0-9_]+" "$WF_DIR" 2>/dev/null \
    | sed "s/^$1\.//" | grep -vx 'GITHUB_TOKEN' | sort -u || true
}

# Test przynależności po dokładnej linii, czysty bash (bez pipe, więc brak kruchości
# pipefail/SIGPIPE w gorącej pętli). $1 = stóg siana z liniami · $2 = igła (nazwy to [A-Z0-9_], bezpieczne jako wzorzec case).
contains_line() {
  case $'\n'"$1"$'\n' in
    *$'\n'"$2"$'\n'*) return 0 ;;
    *) return 1 ;;
  esac
}

# Lista nazw dla danego store, rozróżniając prawdziwy błąd gh (brak admina, repo nie
# znalezione, …) od naprawdę pustego store. Błąd przerywa głośno zamiast po cichu
# raportować wszystko jako "missing" — inaczej błąd uprawnień wygląda identycznie jak
# "nic nie ustawione". Wynik trafia do FETCH_OUT.
FETCH_OUT=""
fetch_names() {
  local noun="$1" raw rc
  raw="$(gh "$noun" list "${REPO_FLAG[@]+"${REPO_FLAG[@]}"}" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "${RED}error:${RESET} could not list ${noun}s on ${REPO_LABEL} — gh said:" >&2
    printf '  %s\n' "$raw" >&2
    if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then
      echo "  ${YELLOW}GH_TOKEN/GITHUB_TOKEN is set${RESET} — it overrides your 'gh auth login' token." >&2
      echo "  ${DIM}A fine-grained PAT without 'Secrets: Read' causes exactly this 403. Either:${RESET}" >&2
      echo "    ${DIM}• run with your keyring token:${RESET}  GH_TOKEN= GITHUB_TOKEN= ${0##*/}" >&2
      echo "    ${DIM}• or grant that PAT 'Secrets: Read' (Variables too, for vars).${RESET}" >&2
    else
      echo "  ${DIM}Listing ${noun}s needs repo admin / 'Secrets: Read'. Wrong repo? Target another with -R owner/repo.${RESET}" >&2
    fi
    exit 2
  fi
  FETCH_OUT="$(printf '%s\n' "$raw" | awk 'NF{print $1}')"
}

# Pobierz oba store i obie listy wymaganych raz, żeby każda sekcja mogła odróżnić "missing"
# od "ustawione w złym kubełku" (workflow czyta je jako inny rodzaj → runtime miss).
fetch_names secret;   HAVE_SECRET="$FETCH_OUT"
fetch_names variable; HAVE_VARIABLE="$FETCH_OUT"
REQ_SECRET="$(discover secrets)"
REQ_VARIABLE="$(discover vars)"
REQ_ALL="$(printf '%s\n%s\n' "$REQ_SECRET" "$REQ_VARIABLE" | grep -v '^$' | sort -u)"

GLOBAL_MISSING=0

check_kind() {
  # $1 = etykieta · $2 = rzeczownik gh (secret|variable)
  local label="$1" noun="$2"
  local required have other other_label name present=0 total=0 missing=0
  if [ "$noun" = "secret" ]; then
    required="$REQ_SECRET"; have="$HAVE_SECRET"; other="$HAVE_VARIABLE"; other_label="variable"
  else
    required="$REQ_VARIABLE"; have="$HAVE_VARIABLE"; other="$HAVE_SECRET"; other_label="secret"
  fi

  echo "${BOLD}${label}${RESET} ${DIM}(required by workflows)${RESET}"
  if [ -z "$required" ]; then
    echo "  ${DIM}— none referenced —${RESET}"
  else
    while IFS= read -r name; do
      [ -z "$name" ] && continue
      total=$((total + 1))
      if contains_line "$have" "$name"; then
        echo "  ${GREEN}✓${RESET} ${name}"
        present=$((present + 1))
      elif contains_line "$other" "$name"; then
        echo "  ${RED}✗${RESET} ${name} ${YELLOW}(set as a ${other_label}, but the workflow reads it as a ${noun} — it won't be picked up)${RESET}"
        missing=$((missing + 1))
      else
        echo "  ${RED}✗${RESET} ${name} ${DIM}(missing)${RESET}"
        missing=$((missing + 1))
      fi
    done <<EOF
$required
EOF
  fi

  # Ustawione w tym kubełku, ale nieodwoływane przez żaden workflow (w żadnym rodzaju) — tylko informacyjnie.
  if [ -n "$have" ]; then
    while IFS= read -r name; do
      [ -z "$name" ] && continue
      if ! contains_line "$REQ_ALL" "$name"; then
        echo "  ${YELLOW}•${RESET} ${name} ${DIM}(set on repo, not used by any workflow)${RESET}"
      fi
    done <<EOF
$have
EOF
  fi

  echo "  ${DIM}${present}/${total} required ${label} set${RESET}"
  echo
  GLOBAL_MISSING=$((GLOBAL_MISSING + missing))
}

echo "${BOLD}StockAgent — GitHub Actions secrets/variables${RESET}"
echo "Repo: ${REPO_LABEL}"
echo
check_kind "Secrets"   secret
check_kind "Variables" variable

TOTAL_REQUIRED="$(printf '%s\n' "$REQ_ALL" | grep -c .)"
if [ "$GLOBAL_MISSING" -gt 0 ]; then
  echo "${RED}${GLOBAL_MISSING} required item(s) missing.${RESET}"
  if [ "$GLOBAL_MISSING" -eq "$TOTAL_REQUIRED" ]; then
    echo "${YELLOW}Nothing is set on ${REPO_LABEL}.${RESET} Secrets/variables are per-repo —"
    echo "  is this the repo your StockAgent workflows actually run on? If it lives on another"
    echo "  repo (e.g. an older one), point the script there:"
    echo "    ${BOLD}scripts/gh-secrets-check.sh -R owner/other-repo${RESET}"
  fi
  echo "Set them from .env with:  ${BOLD}scripts/gh-secrets-sync.sh${RESET}"
  echo "or one-by-one with:       gh secret set <NAME>   /   gh variable set <NAME>"
  exit 1
fi
echo "${GREEN}All required secrets and variables are set.${RESET}"
exit 0
