-- =====================================================================
-- 014_subscribers.sql — odbiorcy spersonalizowanego digestu (U2)
-- Uruchom w Supabase SQL Editor PO 013_idempotency_and_pagination.sql.
--
-- Każdy subskrybent ma własny podzbiór symboli; analiza leci RAZ nad unią
-- wszystkich symboli (bramka volatility bez zmian), a raport jest tylko TNIĘTY
-- per odbiorca. `symbols = '{}'` (pusta tablica) = subskrybent widzi WSZYSTKO
-- (naturalny fallback = dotychczasowy pojedynczy DIGEST_TO_EMAIL).
-- =====================================================================

CREATE TABLE IF NOT EXISTS subscribers (
    id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email   TEXT NOT NULL UNIQUE,
    symbols TEXT[] NOT NULL DEFAULT '{}'
);
