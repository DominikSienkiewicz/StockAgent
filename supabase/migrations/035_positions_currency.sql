-- =====================================================================
-- 035_positions_currency.sql — waluta i prawdziwa data zakupu pozycji (PORT-18)
-- Uruchom PO 034_gate_delta.sql (ręczny workflow „DB Migrate").
--
-- PO CO: tabela `positions` (023) trzyma ilość i średni koszt bez waluty, a jedyną
-- datą jest `as_of` — znacznik świeżości ręcznie utrzymywanej tabeli. Adapter
-- podawał go dotąd jako datę zakupu, więc P/L w PLN nie miał ani właściwej waluty,
-- ani kursu z dnia zakupu. Ta migracja dodaje:
--   * `currency`      — kod ISO 4217 waluty notowania pozycji (3 wielkie litery),
--   * `purchase_date` — prawdziwa data nabycia; NULL = nieznana.
--
-- REGUŁA BACKFILLU WALUTY (wyłącznie wiersze istniejące w chwili dodania kolumny,
-- sufiks tickera bez względu na wielkość liter):
--   * .DE, .AS, .PA, .MI, .MC → 'EUR' (Xetra, Amsterdam, Paryż, Mediolan, Madryt),
--   * .L                      → 'GBP' (Londyn). UWAGA: notowania LSE bywają w pensach
--                               (GBX = 1/100 GBP), a część ETF-ów z .L notuje w USD.
--                               Migracja NICZEGO nie przelicza — gdy źródło ceny podaje
--                               GBX albo USD, popraw walutę importem
--                               (`main_tools.py import-positions`) albo ręcznie.
--   * brak sufiksu i każdy inny → 'USD' (DEFAULT kolumny).
--
-- `purchase_date` istniejących wierszy zostaje NULL: `as_of` to data aktualizacji
-- tabeli, nie zakupu, więc raport mówi wtedy wprost „kurs z dnia zakupu nieznany”
-- zamiast cichego przybliżenia. Prawdziwe daty wnosi import z CSV.
--
-- Odczyt i zapis nowych kolumn idą za flagą `positions_currency_enabled`
-- (przed migracją nowy klucz w zapisie → PGRST204).
--
-- Idempotentne: `currency` powstaje razem z backfillem w jednym bloku, tylko gdy
-- kolumny jeszcze nie ma — ponowny przebieg nie nadpisze walut ustawionych
-- importem. `purchase_date` przez ADD COLUMN IF NOT EXISTS.
-- =====================================================================

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'positions'
          AND column_name = 'currency'
    ) THEN
        ALTER TABLE positions
            ADD COLUMN currency TEXT NOT NULL DEFAULT 'USD'
            CONSTRAINT positions_currency_iso CHECK (currency ~ '^[A-Z]{3}$');

        UPDATE positions
        SET currency = CASE
            WHEN symbol ~* '\.(DE|AS|PA|MI|MC)$' THEN 'EUR'
            WHEN symbol ~* '\.L$' THEN 'GBP'
            ELSE 'USD'
        END;
    END IF;
END
$migration$;

ALTER TABLE positions
    -- Prawdziwa data nabycia lotu; NULL = nieznana (raport nie zgaduje kursu).
    ADD COLUMN IF NOT EXISTS purchase_date DATE;

COMMENT ON COLUMN positions.currency IS
    'Waluta notowania pozycji (ISO 4217). Backfill 035 z sufiksu tickera: .DE/.AS/.PA/.MI/.MC → EUR, .L → GBP (bez przeliczenia GBX), reszta → USD (PORT-18).';
COMMENT ON COLUMN positions.purchase_date IS
    'Data nabycia lotu (kurs NBP z dnia zakupu, PORT-11). NULL = nieznana; as_of to data aktualizacji tabeli, nie zakupu.';
