-- =====================================================================
-- 034_gate_delta.sql — delta, która otworzyła bramkę volatility (EXP-01)
-- Uruchom PO 033_regime_history.sql (ręczny workflow „DB Migrate").
--
-- PO CO: bramka przepuszcza tylko |Δ| ≥ próg, a LLM widzi tę samą zmianę ceny
-- w prompcie („Zmiana od ostatniego cyklu: 3.1%”). Jeśli trend LLM powtarza jej
-- znak (echo momentum) albo ją odwraca, darmowa reguła daje to samo za 0 zł.
-- Test echa (`src.tools.evaluate --echo`) potrzebuje delty przy KAŻDEJ
-- predykcji, a `prediction_logs` jej nie trzymało. Ta migracja dodaje
-- `prediction_logs.gate_delta`: ułamkowa zmiana ceny (0.031 = +3.1%) względem
-- snapshotu poprzedniego cyklu, ta sama wartość, którą sprawdziła bramka.
--
-- BACKFILL (PRZYBLIŻENIE — delta bramki nie była zapisywana):
--   * Bramka liczy deltę względem NAJNOWSZEGO snapshotu ceny symbolu z chwili
--     startu cyklu (`get_last_price` → `state["previous_price"]`), a snapshot
--     cyklu predykcji powstaje dopiero w węźle terminalnym, PO zapisie
--     predykcji. Dlatego referencją jest najnowszy snapshot tego symbolu
--     z `timestamp` ŚCIŚLE sprzed `timestamp` predykcji:
--     `gate_delta = (price_at_prediction − snapshot.price) / snapshot.price`.
--   * Ograniczenia: snapshot zapisany między startem cyklu a zapisem predykcji
--     (nakładający się ręczny run) zostałby wzięty zamiast referencji cyklu;
--     upsert snapshotu w tej samej godzinie (013) podmienia cenę, a zostawia
--     pierwszy `timestamp`; cena bieżąca w bazie jest zaokrąglona do 4 miejsc
--     (DECIMAL(14,4)), więc delta może się różnić od tej z cyklu na ostatnich
--     cyfrach.
--   * Brak poprzedniego snapshotu (pierwszy cykl symbolu, wiersz sprzed 002)
--     albo snapshot z ceną 0 → NULL. Bramka widziała wtedy cold-start
--     (delta 0), nie ruch, więc NULL jest uczciwszy niż 0.
--
-- Zapisuje Fast Loop za flagą `gate_delta_enabled` (przed migracją nowy klucz
-- → PGRST204 i śmierć zapisu całej predykcji). Odczyt w `evaluate --echo`
-- przed migracją kończy się komunikatem o brakującej migracji, nie wyjątkiem.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, backfill tylko po NULL-ach.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Ułamkowa delta ceny, która otworzyła bramkę (EXP-01); NULL = nieznana.
    ADD COLUMN IF NOT EXISTS gate_delta DOUBLE PRECISION;

UPDATE prediction_logs AS p
SET gate_delta = (
    SELECT ((p.price_at_prediction - s.price) / NULLIF(s.price, 0))::DOUBLE PRECISION
    FROM price_snapshots AS s
    WHERE s.symbol = p.symbol
      AND s.timestamp < p.timestamp
    ORDER BY s.timestamp DESC
    LIMIT 1
)
WHERE p.gate_delta IS NULL
  AND p.price_at_prediction IS NOT NULL;

COMMENT ON COLUMN prediction_logs.gate_delta IS
    'Delta ceny vs snapshot poprzedniego cyklu, która otworzyła bramkę (EXP-01). Backfill 034: najnowszy snapshot sprzed predykcji (przybliżenie); NULL = brak referencji.';
