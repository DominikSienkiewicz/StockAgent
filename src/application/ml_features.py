from __future__ import annotations

import pandas as pd

# Kontrakt cech XGBoost — IDENTYCZNY w treningu (widok ml_feature_store) i w
# inference (agent_graph.predict_node). Kolejność musi się zgadzać z widokiem.
ML_FEATURE_COLUMNS = (
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
)

# Kontrakt interwału cechy `price_delta` (krytyczny — train/serve skew):
#   price_delta = (cena_bieżąca - cena_POPRZEDNIEJ_predykcji) / cena_poprzedniej.
# Trening: widok liczy to jako LAG(price_at_prediction) po CAŁEJ prediction_logs
#   (ML-17, migracja 047): także po predykcjach otwartych i oflagowanych —
#   filtry rozliczenia i flag jakości działają dopiero po LAG.
# Inference: predict_node bierze RepositoryPort.get_last_prediction_price (cenę
#   ostatniej zalogowanej predykcji), NIE ostatni snapshot ceny (ten napędza
#   wyłącznie bramkę volatility). Obie ścieżki muszą używać tej samej referencji.
#   ML-18: odczyt pomija wiersz z bieżącej godziny UTC — upsert po
#   (symbol, timestamp_hour) go nadpisze, więc widok nigdy go nie zobaczy.

# Wersja zestawu cech (DAT-09) — rośnie, gdy cecha o TEJ SAMEJ nazwie zmienia
# znaczenie (nowa kolumna to osobna zmiana kontraktu). Historia:
#   1 — `news_volume_24h` liczyło trafne artykuły wśród 50 najnowszych w dowolnym
#       wieku (feed Alpha Vantage bez `time_from`, przed commitem 3738ef2);
#   2 — feed ograniczony do ostatnich 24 h (`time_from` = now − 24 h);
#   3 — feed leniwy per symbol (FIN-04): `news_volume_24h` liczy tylko artykuły
#       z requestu tego symbolu (limit 50 na symbol), a nie trafne artykuły
#       z feedu całej watchlisty scalonego w jednym wsadzie.
# Fast Loop stempluje nią `prediction_logs.feature_set_version` (migracja 027),
# a Slow Loop może trenować wyłącznie na wierszach bieżącej wersji.
FEATURE_SET_VERSION = 3

# ML-14: klucze cech DARMOWYCH w `feature_snapshots.features` (migracja 031).
# Fast Loop zapisuje je w KAŻDYM ukończonym cyklu, także ignored, wyłącznie z
# danych już obecnych w stanie grafu — bez żadnego nowego wywołania zewnętrznego.
# `snapshot_delta` to zmiana względem ostatniego snapshotu ceny (ta, która
# napędza bramkę), a NIE cecha `price_delta` modelu (względem poprzedniej
# predykcji). Cykl za bramką dokłada wektor `ML_FEATURE_COLUMNS` dokładnie
# w postaci podanej modelowi oraz `GATED_SNAPSHOT_EXTRAS`.
FREE_SNAPSHOT_FEATURES = (
    "price",
    "previous_price",
    "snapshot_delta",
    "abs_snapshot_delta",
    "effective_threshold",
    "regime",
    "regime_multiplier",
    "earnings_multiplier",
    "alpha_fusion_score",
    "asset_type",
    "valuation_verdict",
)
GATED_SNAPSHOT_EXTRAS = ("confidence_score", "data_quality_flags")

# Target = ZWROT 12h `(cena_po_12h - cena) / cena`, nie cena bezwzględna.
# Baseline persystencji (cena bez zmian) to wtedy „zwrot = 0", więc bramka
# „pobij baseline" jest sensowna; XGBoost uczy się RUCHU, nie poziomu ceny.
TARGET_COLUMN = "target_return"


# Znacznik czasu wiersza widoku — walk-forward purguje nim horyzont etykiety
# (`src.domain.backtest.purged_train_end`).
TIMESTAMP_COLUMN = "timestamp"


def in_label_time_order(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Wiersze rosnąco po czasie i ich znaczniki (UTC) do purgu horyzontu etykiety.

    Bez kolumny `timestamp` wiersze zostają w kolejności wejścia, a znaczników
    nie ma (`None`) — walk-forward idzie wtedy bez purgu, jak przed decyzją 4.
    """
    if TIMESTAMP_COLUMN not in frame.columns:
        return frame, None
    stamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    ordered = frame.assign(_label_time=stamps).sort_values("_label_time", kind="stable")
    return ordered, ordered["_label_time"]
