from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from src.domain.cross_section import CROSS_SECTION_FEATURE_NAMES
from src.domain.macro_series import MACRO_FEATURE_NAMES
from src.domain.price_features import PRICE_FEATURE_NAMES

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

# Brief W2 (decyzja właściciela): `feature_snapshots` za flagą `market_features_enabled`
# dostaje cechy rynkowe cyklu i wtedy obie ścieżki stemplują wersję 4:
#   4 — jak 3, plus `MARKET_SNAPSHOT_FEATURES` w `feature_snapshots.features`.
# Wektor modelu (`ML_FEATURE_COLUMNS`, widok `ml_feature_store`) się NIE zmienia,
# ale Slow Loop z filtrem wersji trenuje na wierszach wersji z tej samej flagi.
# Flaga off: zestaw i wersja 3 bez zmian.
MARKET_FEATURE_SET_VERSION = 4


def feature_set_version(market_features_enabled: bool) -> int:
    """Wersja zestawu cech cyklu: 4 z cechami rynkowymi, inaczej `FEATURE_SET_VERSION`."""
    return MARKET_FEATURE_SET_VERSION if market_features_enabled else FEATURE_SET_VERSION


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

# DAT-17: flaga świeżego raportu 8-K (1.0 / 0.0 / None = nie wiadomo) dokładana do cech
# darmowych w KAŻDYM cyklu za `corporate_events_enabled`. Nowy klucz, nie zmiana
# znaczenia istniejącego, więc wersja zestawu cech się nie zmienia; wejście modelu
# (`FREE_MODEL_FEATURE_COLUMNS`) go nie bierze, dopóki porównanie go nie uzasadni.
CORPORATE_EVENT_SNAPSHOT_FEATURES = ("fresh_8k",)

# ML-31/ML-32/DAT-22: cechy rynkowe cyklu (wersja 4) liczone przed grafem z darmowych
# źródeł i dokładane do cech darmowych w KAŻDYM cyklu: cenowe symbolu, jego ruch na
# tle uniwersum i zmienne stanu z FRED. Brak danych = None (maska braków).
MARKET_SNAPSHOT_FEATURES: tuple[str, ...] = (
    *PRICE_FEATURE_NAMES,
    *CROSS_SECTION_FEATURE_NAMES,
    *MACRO_FEATURE_NAMES,
)

# ML-52 (decyzja 2026-10-21): Slow Loop trenuje z wektorów `feature_snapshots`
# w wersji zestawu cech 4 (cechy rynkowe ML-31/ML-32/DAT-22).
SNAPSHOT_TRAINING_FEATURE_SET_VERSION = MARKET_FEATURE_SET_VERSION

# ML-52: wejście modelu cech darmowych — wyłącznie BEZWYMIAROWE cechy z wektora cyklu
# (decyzja 4: jeden wspólny model na wielu symbolach). Ceny bezwzględne (`price`,
# `previous_price`) zostają poza modelem; typ aktywa idzie jako dwie cechy 0/1,
# a dzień obserwacji makro (`macro_observed_on`) to metadana, nie cecha.
FREE_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "snapshot_delta",
    "abs_snapshot_delta",
    "effective_threshold",
    "regime_multiplier",
    "earnings_multiplier",
    "alpha_fusion_score",
    "is_crypto",
    "is_etf",
    *PRICE_FEATURE_NAMES,
    *CROSS_SECTION_FEATURE_NAMES,
    *(name for name in MACRO_FEATURE_NAMES if name != "macro_observed_on"),
)

# ML-52: flagi jakości, z którymi cykl nie jest daną treningową — te same co filtr
# widoku `ml_feature_store` (migracja 050: ML-12, EXP-07, DAT-07, DAT-08).
TRAINING_EXCLUDED_QUALITY_FLAGS = frozenset(
    {"llm_analysis_failed", "av_keys_exhausted", "av_unexpected_payload", "av_fetch_error"}
)


def _finite_float(raw: object) -> float | None:
    """Liczba z JSON-a wektora albo None (brak, bool, tekst, NaN/inf)."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def free_model_vector(features: Mapping[str, object]) -> dict[str, float | None]:
    """ML-52: wejście modelu cech darmowych z zapisanego wektora cyklu.

    Brak źródła zostaje brakiem (None → NaN w XGBoost), nie zerem — cisza nie może
    udawać neutralnego rynku. Typ aktywa → `is_crypto` / `is_etf` (akcja = 0/0)."""
    asset_type = features.get("asset_type")
    known_type = isinstance(asset_type, str)
    derived: dict[str, float | None] = {
        "is_crypto": float(asset_type == "CRYPTO") if known_type else None,
        "is_etf": float(asset_type == "ETF") if known_type else None,
    }
    return {
        name: derived[name] if name in derived else _finite_float(features.get(name))
        for name in FREE_MODEL_FEATURE_COLUMNS
    }


def llm_model_vector(features: Mapping[str, object]) -> dict[str, float] | None:
    """ML-52: wektor `ML_FEATURE_COLUMNS` dokładnie w postaci podanej modelowi
    w cyklu za bramką; None, gdy któregoś pola brak (cykl bez wektora modelu)."""
    vector: dict[str, float] = {}
    for name in ML_FEATURE_COLUMNS:
        value = _finite_float(features.get(name))
        if value is None:
            return None
        vector[name] = value
    return vector


def usable_for_llm_model(gated: bool, features: Mapping[str, object]) -> bool:
    """Czy cykl wchodzi do modelu z cechami LLM: za bramką i bez flag jakości,
    które wykluczają go z treningu (`TRAINING_EXCLUDED_QUALITY_FLAGS`)."""
    flags = features.get("data_quality_flags")
    raised = {str(flag) for flag in flags} if isinstance(flags, list) else set()
    return gated and not raised & TRAINING_EXCLUDED_QUALITY_FLAGS


# Target = zwrot do następnego cyklu (~24 h w dni robocze, ~72 h przez weekend)
# `(cena_po_12h - cena) / cena`, nie cena bezwzględna; nazwa `_12h` jest historyczna.
# Baseline persystencji (cena bez zmian) to wtedy „zwrot = 0", więc bramka
# „pobij baseline" jest sensowna; XGBoost uczy się RUCHU, nie poziomu ceny.
TARGET_COLUMN = "target_return"


# Znacznik czasu wiersza widoku — walk-forward purguje nim horyzont etykiety
# (`src.domain.backtest.purged_train_end`).
TIMESTAMP_COLUMN = "timestamp"

# ML-25: chwila rozliczenia wiersza widoku (migracja 050, obok `label_horizon_hours`).
# Dziś tylko do pomiaru nakładania etykiet; purge dalej idzie stałym horyzontem.
SETTLED_AT_COLUMN = "settled_at"


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
