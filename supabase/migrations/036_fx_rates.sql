-- =====================================================================
-- 036_fx_rates.sql — cache kursów NBP z danego dnia (PORT-11)
-- Uruchom PO 035_positions_currency.sql (ręczny workflow „DB Migrate").
--
-- PO CO: P/L portfela w złotych liczy koszt po kursie NBP z dnia zakupu
-- i wartość po kursie bieżącym. Kurs z minionego dnia się nie zmienia, więc
-- zamiast pytać NBP o każdą datę zakupu w każdym cyklu trzymamy go tutaj.
-- NBP jest darmowe — cache oszczędza czas i zależność od dostępności API,
-- nie pieniądze.
--
-- Klucz: (currency, on_date) — dzień, o który pytano. `effective_date` to dzień
-- tabeli A, z której pochodzi kurs: w weekend, święto i przed publikacją tabeli
-- źródło cofa się do poprzedniego dnia roboczego, więc effective_date ≤ on_date.
-- Zapisujemy wyłącznie dni minione (on_date < dziś w Warszawie): kurs „na dziś”
-- przed publikacją tabeli to kurs z wczoraj i po południu byłby nieaktualny.
--
-- Odczyt i zapis idą za flagą `fx_rates_cache_enabled` (przed migracją tabeli
-- nie ma — każde zapytanie kończyłoby się błędem w logu).
--
-- Idempotentne (IF NOT EXISTS).
-- =====================================================================

CREATE TABLE IF NOT EXISTS fx_rates (
    -- Waluta (ISO 4217), o której kurs do PLN pytano.
    currency       TEXT NOT NULL CONSTRAINT fx_rates_currency_iso CHECK (currency ~ '^[A-Z]{3}$'),
    -- Dzień, o który pytano (np. data zakupu pozycji).
    on_date        DATE NOT NULL,
    -- Dzień tabeli NBP, z której pochodzi kurs (cofnięcie do dnia roboczego).
    effective_date DATE NOT NULL,
    -- Kurs średni: PLN za 1 jednostkę waluty. NUMERIC — bez błędu binarnego.
    rate           NUMERIC NOT NULL CONSTRAINT fx_rates_rate_positive CHECK (rate > 0),
    -- Źródło kursu; dziś wyłącznie tabela A NBP.
    source         TEXT NOT NULL DEFAULT 'NBP_A',
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (currency, on_date),
    CONSTRAINT fx_rates_effective_not_after_request CHECK (effective_date <= on_date)
);

COMMENT ON TABLE fx_rates IS
    'Cache kursów NBP (tabela A) z minionych dni do P/L w PLN po kursie z dnia zakupu (PORT-11).';
COMMENT ON COLUMN fx_rates.effective_date IS
    'Dzień tabeli NBP, z której pochodzi kurs; < on_date, gdy on_date nie miał tabeli (weekend, święto).';
