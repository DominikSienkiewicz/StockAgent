#!/usr/bin/env bash
#
# mask-csv-secret.sh — maskuje w logu GitHub Actions każdy element sekretu CSV (BEZ-03).
#
# GitHub maskuje wartość sekretu tylko w całości. Pojedynczy klucz z listy
# `ALPHA_VANTAGE_API_KEYS=k1,k2` (np. w URL-u z tracebacku) wyszedłby więc w logu
# jawnym tekstem. Skrypt rejestruje `::add-mask::` dla każdego elementu: split po
# przecinku i trim, tak jak `Settings` dzieli tę listę. Wołany jako krok PRZED Pythonem.
#
# Usage:   scripts/mask-csv-secret.sh <NAZWA_ZMIENNEJ_ENV>
# Env:     <NAZWA_ZMIENNEJ_ENV> — wartość sekretu (CSV). Czytana z env, nie z argv, żeby
#          nie trafiła do listy procesów. Runner nie drukuje komendy `::add-mask::`
#          w logu; nie dodawaj tu `set -x` ani echo wartości.
#
set -euo pipefail

NAME="${1:?usage: mask-csv-secret.sh <NAZWA_ZMIENNEJ_ENV>}"
VALUE="${!NAME:-}"

while IFS= read -r item; do
  item="${item#"${item%%[![:space:]]*}"}"
  item="${item%"${item##*[![:space:]]}"}"
  if [[ -n "$item" ]]; then
    printf '::add-mask::%s\n' "$item"
  fi
done < <(printf '%s\n' "$VALUE" | tr ',' '\n')
