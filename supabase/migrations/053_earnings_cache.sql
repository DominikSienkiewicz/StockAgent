-- =====================================================================
-- 053_earnings_cache.sql — data najbliższego raportu wyników per symbol
-- (FIN-11). Uruchom workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: bramka earnings potrzebuje daty raportu PRZED bramką volatility, więc
-- Fast Loop pytał Alpha Vantage EARNINGS_CALENDAR o każdy symbol w każdym cyklu
-- (~39 requestów dziennie, jednym kluczem, bez licznika). Teraz Slow Loop odświeża
-- tę tabelę raz w tygodniu wszystkimi kluczami, a Fast Loop czyta wyłącznie ją.
--
-- Kolumny:
--   symbol      — ticker spółki (ETF i krypto nie mają raportów, więc ich nie ma);
--   report_date — najbliższa PRZYSZŁA data raportu w chwili odświeżenia; Fast Loop
--                 liczy z niej `days_until` w dniu cyklu, więc wpis z niedzieli
--                 starcza na cały tydzień;
--   fetched_at  — moment odświeżenia; aplikacja odrzuca wpisy starsze niż
--                 EARNINGS_CACHE_TTL_HOURS (src/domain/earnings.py).
-- Brak nadchodzącego raportu = brak wiersza (adapter AV oddaje None także przy
-- błędzie, więc zapisany „brak” utrwaliłby awarię).
--
-- Odczyt i zapis za flagą `earnings_cache_enabled` (przed migracją tabeli nie ma).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS earnings_cache (
    symbol      TEXT PRIMARY KEY,
    report_date DATE NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Filtr TTL odczytu (`fetched_at >= now() - TTL`).
CREATE INDEX IF NOT EXISTS idx_earnings_cache_fetched_at
    ON earnings_cache (fetched_at);

ALTER TABLE earnings_cache ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE earnings_cache IS
    'Najbliższa data raportu wyników per spółka, odświeżana w Slow Loopie; Fast Loop czyta tylko ją (FIN-11).';
COMMENT ON COLUMN earnings_cache.report_date IS
    'Najbliższa przyszła data raportu w chwili odświeżenia; days_until liczy cykl, który czyta wpis.';
