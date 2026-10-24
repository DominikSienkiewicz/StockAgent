-- =====================================================================
-- 054_council_votes_unique.sql — jeden głos persony na predykcję (NIEZ-14).
-- Uruchom workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: `prediction_logs` jest idempotentne (upsert po symbol + godzina, 013),
-- ale głosy rady szły gołym insertem. Ponowny przebieg Fast Loopa w tej samej
-- godzinie trafiał w ten sam wiersz predykcji i dopisywał drugi komplet głosów,
-- więc predykcja ważyła w rankingu person (015, 018) podwójnie.
--
-- Kolejność jak w 013: najpierw deduplikacja, potem indeks UNIQUE — inaczej
-- istniejące duplikaty wywaliłyby CREATE UNIQUE INDEX błędem 23505. Z każdej
-- pary (prediction_id, investor_name) zostaje wiersz o najwyższym `id`, czyli
-- głos z ostatniego przebiegu (BIGSERIAL rośnie z każdym insertem).
--
-- Kod zapisuje upsertem `on_conflict=prediction_id,investor_name` dopiero za
-- flagą `council_votes_upsert_enabled` — bez tego indeksu PostgREST odrzuca
-- upsert (42P10).
--
-- Idempotentne: DELETE bez duplikatów nic nie usuwa, indeks IF NOT EXISTS.
-- =====================================================================

DELETE FROM council_votes AS older
USING council_votes AS newer
WHERE older.prediction_id = newer.prediction_id
  AND older.investor_name = newer.investor_name
  AND older.id < newer.id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_council_votes_prediction_investor_uniq
    ON council_votes (prediction_id, investor_name);

COMMENT ON INDEX idx_council_votes_prediction_investor_uniq IS
    'Jeden głos persony na predykcję — cel upsertu save_council_votes (NIEZ-14).';
