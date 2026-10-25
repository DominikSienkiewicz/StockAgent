-- =====================================================================
-- 060_model_scorecards_weights_sha256.sql — sha256 pliku wag w karcie
-- modelu (CI-13, runda 3 R3-6).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: wagi idą jako asset Release `model-latest`, nadpisywany co tydzień,
-- a notatki Release opisują tylko ostatnią wersję. Bez hasha w karcie nie da
-- się powiązać metryk przebiegu Slow Loopa z wagami, które ich użyły (ML-22),
-- ani z `model_version` predykcji (sha256 tego samego pliku, WIA-20).
-- Hash liczy `XGBoostAdapter` z bajtów pliku zapisanego na dysku — tego samego,
-- który workflow wysyła do Release (bez ponownej serializacji modelu).
-- NULL = trening bez zapisu wag (odrzucony, pominięty, ewaluacja per symbol),
-- wiersz sprzed tej migracji albo flaga off.
--
-- Zapis kolumny za flagą `model_scorecard_weights_sha256_enabled` (domyślnie
-- off): przed tą migracją nowy klucz to PGRST204 i przepadły wiersz karty.
--
-- Uprawnienia: tabela ma RLS od 038; nowa kolumna tego nie zmienia.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS (CHECK w ADD COLUMN nie dubluje się
-- przy powtórce), COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE model_scorecards
    ADD COLUMN IF NOT EXISTS weights_sha256 TEXT
        CHECK (weights_sha256 ~ '^[0-9a-f]{64}$');

COMMENT ON COLUMN model_scorecards.weights_sha256 IS
    'CI-13: sha256 (hex) pliku wag .ubj zapisanego w tym przebiegu — ten sam plik idzie do Release model-latest, a jego hash to model_version predykcji. NULL = bez zapisu wag, wiersz sprzed 060 albo flaga off.';
