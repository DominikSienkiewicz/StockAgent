-- =====================================================================
-- 033_regime_history.sql — stan histerezy reżimu rynku (ALR-03)
-- Uruchom PO 032_move_interval.sql (ręczny workflow „DB Migrate").
--
-- PO CO: `RegimeDetector.classify()` jest bezstanowe, więc sygnał oscylujący
-- przy progu przełączał mnożnik progu volatility co cykl — szum udający sygnał.
-- Histereza (`apply_hysteresis` w src/domain/regime.py) wchodzi w RISK_OFF od
-- razu, a wychodzi dopiero po N kolejnych surowych odczytach poza RISK_OFF
-- (`regime_exit_confirmations`, domyślnie 2). Potrzebuje pamięci między
-- cyklami: ta tabela trzyma jeden wiersz na cykl Fast Loopa.
--
-- Kolumna `prediction_logs.regime` (017) nie nadaje się na to źródło: ma
-- etykietę reżimu efektywnego tylko przy zapisanej predykcji, a cykl, w którym
-- żaden symbol nie przeszedł bramki, nie zostawia tam nic.
--
--   * raw_regime — surowy wynik `classify()` w tym cyklu,
--   * regime     — reżim efektywny po histerezie (steruje progiem),
--   * streak     — ile kolejnych surowych odczytów poza RISK_OFF potwierdza
--                  wyjście; 0 = brak trwającego wyjścia.
--
-- BEZ BACKFILLU: historia reżimu nie była zapisywana. Pierwszy cykl po
-- włączeniu flagi nie ma wiersza i bierze surowy reżim, jak przed ALR-03.
--
-- Zapisuje Fast Loop za flagą `regime_hysteresis_enabled` (przed migracją
-- tabeli nie ma → błąd PostgREST; cykl loguje go i bierze surowy reżim).
--
-- Idempotentne: CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS.
-- =====================================================================

CREATE TABLE IF NOT EXISTS regime_history (
    id          BIGSERIAL PRIMARY KEY,
    -- Start cyklu Fast Loopa (UTC), który policzył ten stan.
    cycle_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_regime  TEXT NOT NULL
        CHECK (raw_regime IN ('RISK_ON', 'NEUTRAL', 'RISK_OFF')),
    regime      TEXT NOT NULL
        CHECK (regime IN ('RISK_ON', 'NEUTRAL', 'RISK_OFF')),
    streak      INTEGER NOT NULL DEFAULT 0 CHECK (streak >= 0)
);

-- Fast Loop czyta najnowszy wiersz (ORDER BY cycle_at DESC, id DESC LIMIT 1).
CREATE INDEX IF NOT EXISTS idx_regime_history_cycle_at
    ON regime_history (cycle_at DESC);

COMMENT ON TABLE regime_history IS
    'Stan histerezy reżimu rynku, jeden wiersz na cykl Fast Loopa (ALR-03).';
COMMENT ON COLUMN regime_history.regime IS
    'Reżim efektywny po histerezie — to on mnoży próg volatility.';
COMMENT ON COLUMN regime_history.streak IS
    'Kolejne surowe odczyty poza RISK_OFF potwierdzające wyjście; 0 = brak trwającego wyjścia.';
