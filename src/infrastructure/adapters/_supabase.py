"""Wspólna fabryka klienta Supabase (NIEZ-35).

`supabase.create_client` bez opcji daje PostgREST-owi domyślny timeout 120 s, więc jedno
zawieszone zapytanie potrafiło zjeść dwie minuty z 15-minutowego limitu joba. Każdy
adapter bazy (repozytorium predykcji, subskrybenci, pozycje, cache kursów, archiwum
digestów, stan bota) buduje klienta tutaj, z jednym limitem `POSTGREST_TIMEOUT_S`.

Wyjątek to RPC `refresh_ml_feature_store` (REFRESH MATERIALIZED VIEW CONCURRENTLY
w Slow Loopie i backteście): przelicza cały widok cech, więc dostaje osobnego klienta
z dłuższym limitem `FEATURE_STORE_REFRESH_TIMEOUT_S` — tym samym co przed NIEZ-35.
"""

from __future__ import annotations

from supabase import Client, ClientOptions, create_client

POSTGREST_TIMEOUT_S = 15
FEATURE_STORE_REFRESH_TIMEOUT_S = 120


def build_supabase_client(url: str, key: str, *, timeout_s: float = POSTGREST_TIMEOUT_S) -> Client:
    """Klient Supabase z limitem czasu zapytań PostgREST `timeout_s` sekund."""
    return create_client(url, key, options=ClientOptions(postgrest_client_timeout=timeout_s))
