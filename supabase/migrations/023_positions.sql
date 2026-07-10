-- =====================================================================
-- 023_positions.sql — realne pozycje portfela użytkownika (#15)
-- Uruchom w Supabase SQL Editor PO 022_alpha_fusion_score.sql.
--
-- PO CO: cała warstwa ryzyka (stress_test, portfolio_pnl, sizing) liczyła dotąd
-- fikcję RÓWNYCH WAG — user trzymający 40% w NVDA dostawał VaR policzony, jakby
-- miał po 2.3% w 43 tickerach. Ta tabela to ręcznie utrzymywany rejestr realnych
-- pozycji (ilość + średni koszt), z którego domena (`portfolio.py`) liczy realne
-- wagi, P/L i ekspozycję klastra. Adapter Supabase czyta tę tabelę; awaria →
-- `[]` → fallback na dotychczasowe równe wagi (nic się nie psuje).
--
-- `as_of` to znacznik świeżości pozycji — renderer nabija badge STALE, gdy dane
-- są starsze niż N dni (stęchły portfel daje FAŁSZYWY P/L, gorszy niż brak
-- sekcji, więc świeżość musi być widoczna).
--
-- Wzorzec: `014_subscribers.sql` (prosta tabela stanu użytkownika).
-- Idempotentne (IF NOT EXISTS).
-- =====================================================================

CREATE TABLE IF NOT EXISTS positions (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- Ticker pozycji (np. 'NVDA'); wiele wierszy per symbol dozwolone —
    -- najświeższy (max as_of) wygrywa po stronie adaptera.
    symbol     TEXT NOT NULL,
    -- Liczba jednostek; NUMERIC (nie float) — pieniądze/ilości bez błędu binarnego.
    quantity   NUMERIC NOT NULL,
    -- Średni koszt nabycia jednostki; NUMERIC z tego samego powodu co quantity.
    avg_cost   NUMERIC NOT NULL,
    -- Na kiedy pozycja jest aktualna — źródło badge'a STALE w raporcie.
    as_of      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Dostęp: "najświeższe pozycje per symbol" → indeks po (symbol, as_of DESC).
CREATE INDEX IF NOT EXISTS idx_positions_symbol_as_of
    ON positions (symbol, as_of DESC);

COMMENT ON TABLE positions IS
    'Realne pozycje portfela użytkownika (ilość + średni koszt) do wag/P/L/VaR (#15).';
COMMENT ON COLUMN positions.as_of IS
    'Znacznik świeżości pozycji — źródło badge STALE, gdy dane starsze niż N dni.';
