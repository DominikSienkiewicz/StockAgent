"""Shadow mode wariantu promptu albo modelu LLM (EXP-26).

Zmiana promptu oceniona „przed i po” miesza zmianę z rynkiem, a retro-eval na
starych wejściach ma lookahead. Uczciwa ocena to wariant wołany RÓWNOLEGLE
z produkcją na tych samych wejściach: dla stałej podpróbki symboli za bramką
volatility Fast Loop pyta dodatkowo wariant, a jego trend trafia wyłącznie do
`shadow_predictions` (migracja 083) — mail, rada i rekord predykcji są te same.

Podpróbka to sha256(`wariant|symbol`) jako punkt z [0, 1) poniżej `share`:
deterministyczna, ta sama w każdym cyklu i niezależna od listy symboli, więc
porównanie idzie na stałym panelu spółek, wybranym bez ręki autora.

Wariant rozlicza się TĄ SAMĄ regułą co predykcja produkcyjna w reflect
(`Prediction.is_trend_correct` z pasmem SIDEWAYS klasy aktywa), więc różnica
trafności to różnica wariantu, nie miary.

Czysta domena, stdlib.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from src.domain.prediction import Prediction, TrendDirection, sideways_tolerance_for
from src.domain.value_objects import AssetType

# Twardy sufit wywołań wariantu na cykl — niezależnie od konfiguracji.
SHADOW_PROMPT_MAX_PER_CYCLE = 5

# `model_version` wiersza cienia: sha256 hex, jak CHECK w `shadow_predictions` (075).
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_HASH_POINTS = 2**64


def in_shadow_sample(variant: str, symbol: str, share: float) -> bool:
    """Czy symbol należy do stałej podpróbki wariantu o udziale `share` ∈ (0, 1]."""
    if not 0.0 < share <= 1.0:
        raise ValueError(f"shadow sample share must lie in (0, 1], got {share}")
    digest = hashlib.sha256(f"{variant}|{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _HASH_POINTS < share


def variant_version(identity: Mapping[str, str]) -> str:
    """sha256 kanonicznego JSON-a tożsamości wariantu (model, hash szablonu promptu,
    nadpisane opcje) — zmiana któregokolwiek składnika to nowa wersja."""
    canonical = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShadowTrendForecast:
    """Trend i pewność wariantu promptu albo modelu dla jednej predykcji.

    `model_name` — nazwa wariantu z config.toml (wiersz cienia), `model_version` —
    `variant_version` jego tożsamości."""

    model_name: str
    model_version: str
    predicted_trend: TrendDirection
    confidence: float

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("shadow trend forecast needs a variant name")
        if not _SHA256_HEX.match(self.model_version):
            raise ValueError(
                f"shadow trend forecast version must be a sha256 hex, got {self.model_version!r}"
            )
        if not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence}")

    def is_correct(
        self, *, price_at_prediction: Decimal, actual_price: Decimal, asset_type: AssetType | None
    ) -> bool:
        """Trafność kierunku regułą produkcji (reflect): `is_trend_correct` z pasmem
        SIDEWAYS klasy aktywa."""
        call = Prediction(
            symbol="",
            predicted_trend=self.predicted_trend,
            price_at_prediction=price_at_prediction,
            predicted_target_price=price_at_prediction,
        )
        return call.is_trend_correct(actual_price, tolerance=sideways_tolerance_for(asset_type))
