-- migrations/006_fundamentals_cache.sql
-- Cache danych fundamentalnych z Alpha Vantage (OVERVIEW + EARNINGS).
-- Jeden wiersz per symbol — upsert nadpisuje poprzedni snapshot.
-- TTL polityki świeżości żyje w aplikacji (FUNDAMENTALS_CACHE_TTL_HOURS),
-- nie w bazie.

create table if not exists fundamentals_cache (
    symbol         text primary key,
    trailing_pe    numeric,
    forward_pe     numeric,
    peg_ratio      numeric,
    eps_growth_yoy numeric,
    fetched_at     timestamptz not null default now()
);

create index if not exists idx_fundamentals_cache_fetched_at
    on fundamentals_cache (fetched_at);
