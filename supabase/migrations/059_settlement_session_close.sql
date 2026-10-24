-- =====================================================================
-- 059_settlement_session_close.sql — oficjalne zamknięcie sesji jako druga
-- cena rozliczenia predykcji (WIA-61, okres przejściowy).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: `actual_price_after_12h` to cena /quote z chwili cyklu (~05:30 UTC) —
-- nikt z zewnątrz nie odtworzy jej co do centa, więc „dowód” trafności wymaga
-- zaufania do bazy. Decyzja właściciela (opcja „oba”): przez okres przejściowy
-- rozliczenie zapisuje obok niej oficjalne zamknięcie sesji NYSE z datą sesji
-- i źródłem, a mail pokazuje obie trafności. `actual_price_after_12h` zostaje
-- rozliczeniem (trafność, trening, diagnoza bez zmian).
--
--   * `settlement_close_price` — zamknięcie sesji `settlement_session_date`;
--   * `settlement_session_date` — dzień sesji w kalendarzu NYSE (pierwsza sesja
--     po predykcji albo późniejsza; zamknięcia sprzed predykcji nie zapisujemy);
--   * `settlement_source` — pole notowania: `quote_close` (notowanie po
--     zamknięciu sesji, jego cena) albo `quote_previous_close` (notowanie
--     z trwającej sesji, zamknięcie poprzedniej);
--   * `is_trend_correct_session_close` — trafność kierunku na tej cenie (to samo
--     pasmo SIDEWAYS co `is_trend_correct`).
--
-- NULL = brak: wiersze sprzed tej migracji, krypto (handel 24/7, brak sesji),
-- rozliczenie na nieaktualnym notowaniu (DAT-14) i zapis bez flagi. Bez
-- backfillu — notowań starych rozliczeń nie ma w bazie.
--
-- Zapisuje Fast Loop za flagą `session_close_settlement_enabled` (przed
-- migracją nowe klucze → PGRST204 i każdy symbol pada w reflect; preflight
-- FIN-05 zatrzymuje wtedy cykl); odczyt trafności wybiera kolumnę też tylko za
-- flagą. Kolumny w istniejącej tabeli — RLS z 038 je obejmuje, uprawnienia się
-- nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, CHECK w ADD COLUMN (nie dubluje się
-- przy powtórce), COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE prediction_logs
    -- WIA-61: zamknięcie sesji NYSE w skali pozostałych cen (DECIMAL(14,4), ARCH-38).
    ADD COLUMN IF NOT EXISTS settlement_close_price DECIMAL(14, 4),
    ADD COLUMN IF NOT EXISTS settlement_session_date DATE,
    ADD COLUMN IF NOT EXISTS settlement_source TEXT
        CHECK (settlement_source IN ('quote_close', 'quote_previous_close')),
    ADD COLUMN IF NOT EXISTS is_trend_correct_session_close BOOLEAN;

COMMENT ON COLUMN prediction_logs.settlement_close_price IS
    'WIA-61: oficjalne zamknięcie sesji NYSE z dnia settlement_session_date — druga cena '
    'rozliczenia obok actual_price_after_12h (/quote). NULL = brak (krypto, sprzed 059).';

COMMENT ON COLUMN prediction_logs.settlement_session_date IS
    'WIA-61: dzień sesji NYSE, której zamknięcie zapisano (pierwsza sesja po predykcji '
    'albo późniejsza) — z nim cenę sprawdzi każdy publiczny serwis.';

COMMENT ON COLUMN prediction_logs.settlement_source IS
    'WIA-61: pole notowania zamknięcia: quote_close (notowanie po zamknięciu sesji) '
    'albo quote_previous_close (zamknięcie poprzedniej sesji z notowania w trakcie sesji).';

COMMENT ON COLUMN prediction_logs.is_trend_correct_session_close IS
    'WIA-61: trafność kierunku na settlement_close_price (to samo pasmo SIDEWAYS co '
    'is_trend_correct). NULL = brak zamknięcia sesji.';
