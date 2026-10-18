#!/usr/bin/env bash
#
# notify-failure.sh — mail o porażce albo anulowaniu jobu GitHub Actions przez Resend (NIEZ-23).
#
# Wołany z ostatniego kroku workflowu z `if: failure() || cancelled()`. Czysty bash z curl
# i jq (bez Pythona), więc działa także wtedy, gdy padło `uv sync`. Klucz Resend idzie do
# curl przez stdin (`curl -K -`), nigdy przez argv. Własny błąd skryptu nie zmienia wyniku
# jobu: kończy się adnotacją ::warning:: i kodem 0.
#
# Usage:   scripts/notify-failure.sh "<nazwa workflowu>"
# Env:     RESEND_API_KEY, DIGEST_TO_EMAIL (sekrety), RUN_URL, JOB_STATUS (failure|cancelled).
#          Nadawca z `digest_from_email` w config.toml.
#
set -uo pipefail

WORKFLOW="${1:-workflow}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

warn() { echo "::warning::notify-failure: $*"; exit 0; }

[[ -n "${RESEND_API_KEY:-}" && -n "${DIGEST_TO_EMAIL:-}" ]] \
  || warn "brak RESEND_API_KEY albo DIGEST_TO_EMAIL — alert nie wyszedł"
command -v jq >/dev/null 2>&1 || warn "brak jq — alert nie wyszedł"

FROM="$(sed -nE 's/^digest_from_email[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$ROOT/config.toml" 2>/dev/null | head -n1)"
[[ -n "$FROM" ]] || FROM="onboarding@resend.dev"
STATUS="${JOB_STATUS:-failure}"
URL="${RUN_URL:-}"

BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT
jq -n --arg from "$FROM" --arg to "$DIGEST_TO_EMAIL" --arg wf "$WORKFLOW" --arg st "$STATUS" --arg url "$URL" '{
  from: $from,
  to: [$to],
  subject: "⚠️ StockAgent: \($wf) — \($st)",
  text: "Workflow \($wf) zakończył się wynikiem: \($st).\nPrzebieg: \($url)",
  html: "<p>Workflow <strong>\($wf)</strong> zakończył się wynikiem: <strong>\($st)</strong>.</p><p><a href=\"\($url)\">Otwórz przebieg</a></p>"
}' >"$BODY" || warn "nie udało się zbudować treści alertu"

printf 'header = "Authorization: Bearer %s"\n' "$RESEND_API_KEY" | curl -K - --silent --show-error --fail --max-time 20 -H 'Content-Type: application/json' --data-binary @"$BODY" https://api.resend.com/emails >/dev/null \
  || warn "Resend nie przyjął alertu"
echo "notify-failure: alert wysłany ($WORKFLOW, $STATUS)"
