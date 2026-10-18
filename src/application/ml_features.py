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
# Trening: widok liczy to jako LAG(price_at_prediction) po prediction_logs.
# Inference: predict_node bierze RepositoryPort.get_last_prediction_price (cenę
#   ostatniej zalogowanej predykcji), NIE ostatni snapshot ceny (ten napędza
#   wyłącznie bramkę volatility). Obie ścieżki muszą używać tej samej referencji.

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
