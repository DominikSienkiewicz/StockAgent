#!/usr/bin/env bash
#
# prod-state-check.sh — jednorazowy odczyt stanu produkcji StockAgenta (NIEZ-47):
# GitHub (sekrety po nazwie, workflowy, ostatnie przebiegi, ochrona gałęzi, Pages,
# wynik ostatniego Slow Loopa) i Supabase przez REST API (PostgREST) tym samym
# kluczem, którego używa aplikacja.
#
# Read-only z konstrukcji (pilnuje tests/test_workflow_policy.py):
# - GitHub: wyłącznie `gh ... list`, `gh run view` i `gh api` bez metody i pól;
# - Supabase: wyłącznie `curl --get`, bez funkcji RPC — obecność migracji od 017
#   wynika z opisu OpenAPI (tabele, kolumny, funkcje), a nie z ich wywołania.
#   Migracje, które zmieniają tylko definicję widoku, indeks, ograniczenie albo
#   uprawnienia, OpenAPI nie pokazuje — wynik opisuje je jako nie do sprawdzenia.
# Klucz idzie do curl przez stdin (`curl -K -`), nigdy przez argv ani na wyjście;
# z JWT wypisywany jest wyłącznie claim `role`.
#
# Wynik (Markdown na stdout) zawiera dane wrażliwe (nazwy sekretów, liczbę
# subskrybentów) — zapisuj go wyłącznie do gitignorowanego docs/concepts/:
#   scripts/prod-state-check.sh > docs/concepts/prod-state-$(date +%F).md
#
# Usage:   scripts/prod-state-check.sh [-R owner/repo] [-f .env] [--skip-github] [--skip-supabase]
# Env:     SUPABASE_URL, SUPABASE_KEY, ALPHA_VANTAGE_API_KEYS — ze środowiska albo z pliku
#          env (domyślnie <repo>/.env); zmienna środowiskowa wygrywa z plikiem.
# Exit:    0 = odczyt zakończony (nieudane pojedyncze odczyty są opisane w wyniku)
#          2 = błąd konfiguracji (brak narzędzia, logowania gh albo kluczy)
#
set -euo pipefail

REPO=""
ENV_FILE=""
RUN_GITHUB=1
RUN_SUPABASE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -R|--repo) REPO="${2:-}"; shift 2 ;;
    -f|--file) ENV_FILE="${2:-}"; shift 2 ;;
    --skip-github) RUN_GITHUB=0; shift ;;
    --skip-supabase) RUN_SUPABASE=0; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF_DIR="$ROOT/.github/workflows"
[[ -n "$ENV_FILE" ]] || ENV_FILE="$ROOT/.env"
MIN_SAMPLES="$(grep -oE 'MIN_SAMPLES_FOR_TRAINING = [0-9]+' "$ROOT/src/application/use_cases/train_model.py" 2>/dev/null | grep -oE '[0-9]+$' || true)"

# Obiekty, po których rozpoznajemy migrację: numer, rodzaj, tabela/funkcja, kolumna.
# Rodzaj `opaque` to migracja bez śladu w OpenAPI (nazwa obiektu, powód): sama
# definicja widoku, indeks, ograniczenie albo uprawnienia. Tests/test_prod_state_probes.py
# pilnuje, żeby każda migracja od 017 miała dokładnie jedną sondę z obiektem, który
# ta migracja naprawdę tworzy.
MIGRATION_PROBES="017 column prediction_logs regime
018 function persona_accuracy_stats -
019 table model_scorecards -
020 column prediction_logs decision_receipts
021 table shock_alerts -
022 column prediction_logs alpha_fusion_score
023 table positions -
024 column prediction_logs revealed_at
025 column prediction_logs edge_sigma
026 opaque ml_feature_store view-definition
027 column prediction_logs feature_set_version
028 column model_scorecards drift_baseline_rmse
029 column prediction_logs settled_at
030 column prediction_logs recipe_version
031 table feature_snapshots -
032 column prediction_logs interval_low
033 table regime_history -
034 column prediction_logs gate_delta
035 column positions currency
036 table fx_rates -
037 column prediction_logs experiment_arms
038 opaque row_level_security rls
039 opaque ml_feature_store view-definition
040 table llm_calls -
041 table cycle_runs -
042 table reference_predictions -
043 column prediction_logs commitment_payload
044 column prediction_logs calibration_score_version
045 function persona_vote_outcomes -
046 column prediction_logs correction_category
047 opaque ml_feature_store view-definition
048 opaque prediction_logs_predicted_trend_known constraint
049 column model_scorecards run_id
050 column ml_feature_store label_horizon_hours
051 column prediction_logs trend_probabilities
052 column model_scorecards feature_stats
053 table earnings_cache -
054 opaque idx_council_votes_prediction_investor_uniq index
055 column prediction_logs stale_quote
056 column fundamentals_cache market_cap
057 table price_level_alerts -
058 opaque ml_feature_store view-definition
059 column prediction_logs settlement_close_price
060 column model_scorecards weights_sha256
061 table digests -
062 table corporate_actions -
063 function match_news_embeddings_exact -
064 column model_scorecards dataset_sha256
065 function persona_scoring_stats -
070 opaque match_news_embeddings rls
071 function schema_version -
072 column prediction_logs entry_price
073 table paper_trades -
074 column prediction_logs invalidation_price
075 table shadow_predictions -
076 function match_news_embeddings_pit -"

# Zakres z listy sond, żeby nagłówek nie zestarzał się z kolejną migracją.
PROBE_RANGE="$(awk 'NR == 1 {first = $1} {last = $1} END {print first "–" last}' <<<"$MIGRATION_PROBES")"

SB_URL=""
SB_KEY=""

die() { echo "error: $*" >&2; exit 2; }

# Odczyt wartości pojedynczego klucza z pliku env (ostatnie przypisanie wygrywa): wszystko po
# pierwszym '=', minus jedna warstwa otaczających cudzysłowów i końcowy CR.
get_env_value() {
  local key="$1" file="$2" line val
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$file" | tail -n1 || true)"
  [[ -z "$line" ]] && return 1
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line#export }"
  val="${line#*=}"
  val="${val%$'\r'}"
  case "$val" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
    *) ;;
  esac
  printf '%s' "$val"
}

# Wartość konfiguracji: zmienna środowiskowa, a gdy pusta — plik env.
config_value() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then
    printf '%s' "${!key}"
    return 0
  fi
  [[ -f "$ENV_FILE" ]] || return 1
  get_env_value "$key" "$ENV_FILE"
}

join_lines() { paste -sd ',' - | sed 's/,/, /g'; }

row() { echo "| $1 | ${2:-błąd odczytu albo brak danych} |"; }

# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

github_section() {
  command -v gh >/dev/null 2>&1 || die "GitHub CLI (gh) not found — install from https://cli.github.com"
  gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"
  local repo="$REPO"
  [[ -n "$repo" ]] || repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

  echo "## GitHub ($repo)"
  echo
  echo "### Sekrety (tylko nazwy)"
  echo
  local have required missing
  have="$(gh secret list -R "$repo" | awk 'NF{print $1}' | sort -u)"
  required="$(grep -rhoE 'secrets\.[A-Za-z0-9_]+' "$WF_DIR" | sed 's/^secrets\.//' | grep -vx 'GITHUB_TOKEN' | sort -u || true)"
  missing="$(comm -23 <(printf '%s\n' "$required") <(printf '%s\n' "$have") | grep -v '^$' || true)"
  echo "- Ustawione: $(printf '%s\n' "$have" | join_lines)"
  if [[ -n "$missing" ]]; then
    echo "- **Brakuje** (czytane przez workflowy z tego checkoutu): $(printf '%s\n' "$missing" | join_lines)"
  else
    echo "- Wszystkie sekrety czytane przez workflowy są ustawione."
  fi
  echo

  echo "### Workflowy"
  echo
  echo "| Workflow | Stan | Ostatni przebieg (UTC · zdarzenie · wynik · commit) |"
  echo "|---|---|---|"
  local path state wf last
  while IFS=$'\t' read -r path state; do
    wf="$(basename "$path")"
    last="$(gh run list -R "$repo" --workflow "$wf" --limit 1 \
      --json createdAt,event,conclusion,headSha \
      -q '.[0] // empty | "\(.createdAt) · \(.event) · \(.conclusion) · \(.headSha[0:7])"' || true)"
    echo "| \`$wf\` | $state | ${last:-—} |"
  done < <(gh workflow list -R "$repo" --all --json path,state -q '.[] | "\(.path)\t\(.state)"')
  echo

  echo "### Repo"
  echo
  if gh api "repos/$repo/branches/main/protection" >/dev/null 2>&1; then
    echo "- \`main\`: branch protection włączona"
  else
    echo "- \`main\`: brak branch protection (albo brak uprawnień do odczytu)"
  fi
  # Przy 404 `gh api` wypisuje ciało błędu na stdout — wtedy wynik odrzucamy.
  local rulesets pages
  rulesets="$(gh api "repos/$repo/rulesets" -q 'length' 2>/dev/null)" || rulesets="?"
  echo "- Rulesety: $rulesets"
  pages="$(gh api "repos/$repo/pages" -q '.html_url' 2>/dev/null)" || pages=""
  echo "- GitHub Pages: ${pages:-nieskonfigurowane}"
  echo

  slow_loop_summary "$repo"
}

slow_loop_summary() {
  local repo="$1" run_id log statuses max_samples recap
  echo "### Ostatni Slow Loop"
  echo
  run_id="$(gh run list -R "$repo" --workflow slow_loop_weekly.yml --limit 1 --json databaseId -q '.[0].databaseId // empty' || true)"
  if [[ -z "$run_id" ]]; then
    echo "- Brak przebiegów."
    echo
    return 0
  fi
  log="$(gh run view -R "$repo" "$run_id" --log 2>/dev/null || true)"
  statuses="$(grep -oE ": \{'status': '[a-z_]+'" <<<"$log" | sed -E "s/.*'([a-z_]+)'$/\1/" | sort | uniq -c \
    | awk '{printf "%s%s × %s", (NR > 1 ? ", " : ""), $2, $1}' || true)"
  max_samples="$(grep -oE "Not enough samples for '[^']+' \([0-9]+ <" <<<"$log" | grep -oE '\([0-9]+' | tr -d '(' | sort -n | tail -1 || true)"
  if grep -q 'Weekly recap failed' <<<"$log"; then
    recap="**błąd** (Weekly recap failed)"
  else
    recap="bez błędu w logu"
  fi
  echo "- Przebieg: $run_id"
  echo "- Wynik treningu per symbol: ${statuses:-brak linii statusu w logu}"
  echo "- Najwięcej próbek wśród pominiętych: ${max_samples:-—} (próg: ${MIN_SAMPLES:-?})"
  echo "- Weekly recap: $recap"
  echo
}

# ---------------------------------------------------------------------------
# Supabase (PostgREST, wyłącznie GET)
# ---------------------------------------------------------------------------

# GET do PostgREST; klucz przez stdin (`curl -K -`), nigdy w argv. Dodatkowe opcje curl w "$@".
rest_get() {
  local path="$1"; shift
  printf 'header = "apikey: %s"\nheader = "Authorization: Bearer %s"\n' "$SB_KEY" "$SB_KEY" | curl -K - --get --silent --show-error --fail "$@" "$SB_URL/rest/v1/$path"
}

# Liczba wierszy z nagłówka Content-Range (Prefer: count=exact), bez pobierania danych.
rest_count() {
  rest_get "$1&limit=1" -H 'Prefer: count=exact' -o /dev/null -D - \
    | tr -d '\r' | awk -F/ 'tolower($1) ~ /^content-range/ {print $2}'
}

# Kod HTTP odczytu bez ciała — sonda istnienia tabeli albo kolumny (GET z limit=0).
rest_status() {
  printf 'header = "apikey: %s"\nheader = "Authorization: Bearer %s"\n' "$SB_KEY" "$SB_KEY" | curl -K - --get --silent --output /dev/null --write-out '%{http_code}' "$SB_URL/rest/v1/$1"
}

# Najnowsza wartość kolumny; pusta tabela (albo wiersze ukryte przez RLS) to „brak wierszy”.
rest_latest() {
  local table="$1" column="$2" body value
  body="$(rest_get "$table?select=$column&order=$column.desc&limit=1")" || return 1
  value="$(jq -r --arg c "$column" '.[0][$c] // empty' <<<"$body")"
  printf '%s' "${value:-brak wierszy}"
}

# Rola klucza bez ujawniania go: nowe klucze Supabase mają czytelny prefiks,
# a ze starego JWT dekodujemy wyłącznie claim `role`.
key_role() {
  case "$1" in
    sb_secret_*) echo "secret (nowy format, uprawnienia jak service_role)"; return 0 ;;
    sb_publishable_*) echo "publishable (nowy format, odpowiednik anon)"; return 0 ;;
    *) ;;
  esac
  printf '%s' "$1" | cut -d. -f2 \
    | jq -Rr 'gsub("-"; "+") | gsub("_"; "/") | . + ("=" * ((4 - length % 4) % 4)) | @base64d | fromjson | .role // "brak claimu role"' 2>/dev/null \
    || echo "nie da się odczytać (klucz nie jest JWT)"
}

# Opis migracji, której OpenAPI nie pokaże: etykieta wiersza i werdykt.
opaque_label() {
  local name="$1" reason="$2"
  case "$reason" in
    view-definition) echo "definicja widoku \`$name\`" ;;
    index) echo "indeks \`$name\`" ;;
    constraint) echo "ograniczenie \`$name\`" ;;
    rls) echo "RLS i uprawnienia ról API (\`$name\`)" ;;
    *) echo "obiekt \`$name\`" ;;
  esac
}

OPAQUE_VERDICT="n/d (poza zasięgiem OpenAPI)"

opaque_note() {
  echo
  echo "Wiersze n/d to migracje, które zmieniają tylko definicję widoku, indeks, ograniczenie albo uprawnienia: OpenAPI nie pokazuje takich obiektów, więc ich obecność potwierdzi dashboard Supabase albo \`supabase migration list\`. Migracja 026 zmienia wyłącznie definicję widoku \`ml_feature_store\`, a 038 włącza RLS."
}

migrations_from_openapi() {
  local spec="$1" num kind name column present label
  echo "| Migracja | Obiekt | Jest w bazie |"
  echo "|---|---|---|"
  while read -r num kind name column; do
    case "$kind" in
      table)
        present="$(jq -r --arg t "$name" '.definitions[$t] != null' <<<"$spec")"
        label="tabela \`$name\`" ;;
      column)
        present="$(jq -r --arg t "$name" --arg c "$column" '.definitions[$t].properties[$c] != null' <<<"$spec")"
        label="kolumna \`$name.$column\`" ;;
      opaque)
        echo "| $num | $(opaque_label "$name" "$column") | $OPAQUE_VERDICT |"
        continue ;;
      *)
        present="$(jq -r --arg f "/rpc/$name" '.paths[$f] != null' <<<"$spec")"
        label="funkcja \`$name\`" ;;
    esac
    if [[ "$present" == "true" ]]; then present="tak"; else present="**nie**"; fi
    echo "| $num | $label | $present |"
  done <<<"$MIGRATION_PROBES"
}

# Bez OpenAPI (np. klucz publishable): sonda GET per tabela/kolumna. Funkcji nie da się
# sprawdzić bez ich wywołania, więc zostają n/d.
migrations_from_probes() {
  local num kind name column code label verdict
  echo "| Migracja | Obiekt | Jest w bazie |"
  echo "|---|---|---|"
  while read -r num kind name column; do
    case "$kind" in
      table) label="tabela \`$name\`"; code="$(rest_status "$name?select=*&limit=0" || true)" ;;
      column) label="kolumna \`$name.$column\`"; code="$(rest_status "$name?select=$column&limit=0" || true)" ;;
      opaque)
        echo "| $num | $(opaque_label "$name" "$column") | $OPAQUE_VERDICT |"
        continue ;;
      *) label="funkcja \`$name\`"; code="" ;;
    esac
    case "$code" in
      200|206) verdict="tak" ;;
      400|404) verdict="**nie** (HTTP $code)" ;;
      401|403) verdict="brak dostępu dla tego klucza (HTTP $code)" ;;
      "") verdict="n/d bez OpenAPI (funkcji nie wywołujemy)" ;;
      *) verdict="? (HTTP $code)" ;;
    esac
    echo "| $num | $label | $verdict |"
  done <<<"$MIGRATION_PROBES"
}

# Ile rozliczonych predykcji ma niepustą listę `data_quality_flags` — „X z Y (Z%)”.
settled_quality_flags() {
  rest_get "prediction_logs?select=data_quality_flags&actual_price_after_12h=not.is.null&limit=10000" \
    | jq -r 'length as $all | map(select((.data_quality_flags // []) | length > 0)) | length as $flagged
      | if $all == 0 then "brak rozliczonych" else "\($flagged) z \($all) (\($flagged * 100 / $all | floor)%)" end'
}

# Liczba wierszy per symbol w ml_feature_store (to próbki treningowe), stronicowana po 1000.
feature_store_counts() {
  local offset=0 page rows="[]" n
  while :; do
    page="$(rest_get "ml_feature_store?select=symbol&order=symbol&limit=1000&offset=$offset")" || return 1
    n="$(jq 'length' <<<"$page")"
    rows="$(jq -s 'add' <<<"$rows $page")"
    [[ "$n" -lt 1000 ]] && break
    offset=$((offset + 1000))
  done
  local threshold="${MIN_SAMPLES:-0}"
  jq -r --argjson min "$threshold" '
    group_by(.symbol) | map({s: .[0].symbol, n: length}) | sort_by(-.n) as $c
    | "- Łącznie: \(map(.n) | add // 0) wierszy, \($c | length) symboli, symboli z co najmniej \($min) próbkami: \([$c[] | select(.n >= $min)] | length)",
      "",
      "| Symbol | Próbki |",
      "|---|---:|",
      ($c[] | "| \(.s) | \(.n) |")' <<<"$rows"
}

supabase_section() {
  command -v curl >/dev/null 2>&1 || die "curl not found"
  command -v jq >/dev/null 2>&1 || die "jq not found — brew install jq"
  SB_URL="$(config_value SUPABASE_URL || true)"
  SB_KEY="$(config_value SUPABASE_KEY || true)"
  [[ -n "$SB_URL" && -n "$SB_KEY" ]] || die "SUPABASE_URL / SUPABASE_KEY not set (environment or $ENV_FILE)"
  SB_URL="${SB_URL%/}"

  echo "## Supabase (REST, tylko GET)"
  echo
  local role spec
  role="$(key_role "$SB_KEY")"
  echo "- Rola klucza \`SUPABASE_KEY\`: $role"
  case "$role" in
    publishable*|anon)
      echo "- Uwaga: klucz anon/publishable widzi tylko to, na co pozwalają uprawnienia tabel i RLS. Pełne odczyty poniżej znaczą, że tabele są dla niego otwarte (BEZ-04, BEZ-05)." ;;
    *) ;;
  esac
  echo "- Wersja Postgresa i plan projektu: n/d przez REST — odczytaj w dashboardzie Supabase."
  echo

  echo "### Migracje $PROBE_RANGE"
  echo
  if spec="$(rest_get "" -H 'Accept: application/openapi+json' 2>/dev/null)"; then
    echo "Źródło: opis OpenAPI."
    echo
    migrations_from_openapi "$spec"
    opaque_note
  else
    echo "Źródło: sondy GET z \`limit=0\` (OpenAPI niedostępne dla tego klucza)."
    echo
    migrations_from_probes
    opaque_note
  fi
  echo

  echo "### Dane"
  echo
  echo "| Odczyt | Wartość |"
  echo "|---|---|"
  row "Predykcje w \`prediction_logs\`" "$(rest_count 'prediction_logs?select=symbol' || true)"
  row "Predykcje z \`revealed_at\`" "$(rest_count 'prediction_logs?select=symbol&revealed_at=not.is.null' || true)"
  row "Ostatnia predykcja (\`prediction_logs.timestamp\`)" "$(rest_latest prediction_logs timestamp || true)"
  row "Ostatni snapshot ceny (\`price_snapshots.timestamp\`)" "$(rest_latest price_snapshots timestamp || true)"
  row "Ostatni odczyt fundamentów (\`fundamentals_cache.fetched_at\`)" "$(rest_latest fundamentals_cache fetched_at || true)"
  row "Subskrybenci (\`subscribers\`)" "$(rest_count 'subscribers?select=*' || true)"
  row "Rozliczone predykcje z flagą jakości danych (EXP-07)" "$(settled_quality_flags || true)"
  echo

  echo "### Próbki treningowe w \`ml_feature_store\` (próg: ${MIN_SAMPLES:-?})"
  echo
  feature_store_counts || echo "- Błąd odczytu \`ml_feature_store\`."
  echo

  echo "### Klucze Alpha Vantage"
  echo
  local av count
  av="$(config_value ALPHA_VANTAGE_API_KEYS || true)"
  count="$(printf '%s' "$av" | tr ',' '\n' | grep -c '[^[:space:]]' || true)"
  echo "- W lokalnym pliku env: ${count:-0}. Wartości sekretu w GitHubie nie da się odczytać — porównaj u siebie."
  echo
}

echo "# Stan produkcji StockAgent — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo
echo "_Wygenerowane przez \`scripts/prod-state-check.sh\` (read-only). Zawiera dane wrażliwe — trzymaj w gitignorowanym \`docs/concepts/\`._"
echo
if [[ "$RUN_GITHUB" -eq 1 ]]; then github_section; fi
if [[ "$RUN_SUPABASE" -eq 1 ]]; then supabase_section; fi
exit 0
