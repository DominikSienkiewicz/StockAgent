-- =====================================================================
-- 042_reference_predictions.sql — darmowy sygnał referencyjny (WIA-24).
-- Uruchom PO 041_cycle_runs.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: pre-rejestrowana reguła momentum tygodniowego (docs/reference-protocol-v1.md)
-- daje żywy track record za 0 zł podczas pauzy płatnej pętli i jest benchmarkiem,
-- który LLM musi pobić. Odczyt, wywołanie i rozliczenie żyją w JEDNYM wierszu tej
-- tabeli, a nie w `prediction_logs`: tamtą tabelę czytają ścieżki LLM (reflect
-- z płatną diagnozą, RAG, widok `ml_feature_store` z LAG po wszystkich rozliczonych
-- wierszach, track record, recap, kalibracja), a upsert po (symbol, timestamp_hour)
-- scaliłby wiersz referencyjny z predykcją LLM z tej samej godziny.
--
-- Kolumny:
--   symbol, read_at, price — własny odczyt darmowego /quote w przebiegu Slow Loopa
--                 (nigdy `price_snapshots`); jeden odczyt na symbol i chwilę;
--   predicted_trend — wywołanie z tego odczytu: BULLISH / BEARISH, NULL = abstynencja
--                 (brak odczytu sprzed 6–8 dni albo cena bez zmian);
--   recipe_version — wersja protokołu, 'reference-v1'; CHECK nie wpuszcza tu
--                 przepisu LLM;
--   outcome, settled_price, settled_at — rozliczenie wywołania odczytem następnego
--                 przebiegu: HIT / MISS z ceną rozliczenia albo EXPIRED bez ceny.
--
-- Zapis i odczyt za flagą `reference_signal_enabled` (przed migracją tabeli nie ma).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta. Granty dla anon / authenticated
-- odbierają domyślne uprawnienia ustawione w 038.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS i COMMENT można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS reference_predictions (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    read_at         TIMESTAMPTZ NOT NULL,
    price           NUMERIC NOT NULL CONSTRAINT reference_predictions_price_positive
                        CHECK (price > 0),
    predicted_trend TEXT CONSTRAINT reference_predictions_trend_known
                        CHECK (predicted_trend IN ('BULLISH', 'BEARISH')),
    recipe_version  TEXT NOT NULL DEFAULT 'reference-v1'
                        CONSTRAINT reference_predictions_reference_label
                        CHECK (recipe_version ~ '^reference-v[0-9]+$'),
    outcome         TEXT CONSTRAINT reference_predictions_outcome_known
                        CHECK (outcome IN ('HIT', 'MISS', 'EXPIRED')),
    settled_price   NUMERIC CONSTRAINT reference_predictions_settled_price_positive
                        CHECK (settled_price > 0),
    settled_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT reference_predictions_one_read_per_symbol_and_time
        UNIQUE (symbol, read_at),
    -- Rozliczyć można tylko wywołanie, nie abstynencję.
    CONSTRAINT reference_predictions_outcome_needs_call
        CHECK (outcome IS NULL OR predicted_trend IS NOT NULL),
    -- Wynik i chwila rozliczenia pojawiają się razem.
    CONSTRAINT reference_predictions_outcome_has_time
        CHECK ((outcome IS NULL) = (settled_at IS NULL)),
    -- HIT / MISS mają cenę rozliczenia, EXPIRED i wiersz otwarty — nie.
    CONSTRAINT reference_predictions_price_only_when_judged
        CHECK (
            CASE WHEN outcome IN ('HIT', 'MISS')
                THEN settled_price IS NOT NULL
                ELSE settled_price IS NULL
            END
        )
);

-- Odczyt bazy tygodnia: wszystkie odczyty od chwili (przebieg − 8 dni).
CREATE INDEX IF NOT EXISTS idx_reference_predictions_read_at
    ON reference_predictions (read_at);

-- Otwarte wywołania do rozliczenia w kolejnym przebiegu.
CREATE INDEX IF NOT EXISTS idx_reference_predictions_open_calls
    ON reference_predictions (read_at)
    WHERE predicted_trend IS NOT NULL AND outcome IS NULL;

ALTER TABLE reference_predictions ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE reference_predictions IS
    'Sygnał referencyjny WIA-24: odczyt darmowego /quote, wywołanie reguły i jego rozliczenie w jednym wierszu; poza prediction_logs.';
COMMENT ON COLUMN reference_predictions.predicted_trend IS
    'BULLISH / BEARISH ze znaku zmiany od odczytu sprzed 6–8 dni; NULL = abstynencja.';
COMMENT ON COLUMN reference_predictions.recipe_version IS
    'Wersja protokołu (docs/reference-protocol-v1.md = reference-v1); zmiana reguły = nowa wersja.';
COMMENT ON COLUMN reference_predictions.outcome IS
    'HIT / MISS rozliczone odczytem z 6–8 dni po wywołaniu, EXPIRED bez takiego odczytu; NULL = otwarte albo abstynencja.';
