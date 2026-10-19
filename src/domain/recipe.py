"""Przepis predykcji (WIA-20) — z czego powstała prognoza i jego wersja.

Track record bez wersji przepisu miesza wyniki różnych promptów, modeli
i składów rady: zmiana przepisu wygląda wtedy jak zmiana umiejętności, a liczba
prób do Deflated Sharpe jest nieznana. `PredictionRecipe` zbiera składniki
przepisu jako gotowe odciski (hashe liczy DI — domena nie czyta plików),
a `version` to jeden sha256 do segmentacji wyników.

Tożsamość przebiegu (`git_sha`, `run_id`) jest w przepisie dla audytu, ale NIE
wchodzi do `version`: ten sam przepis w kolejnych przebiegach to wciąż jeden
segment, a commit bez wpływu na prognozę (docs, testy) nie zakłada nowego.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from functools import cached_property

from src.domain.equity_curve import DirectionalHitRate, directional_hit_rate

# Segment wierszy bez wersji (sprzed WIA-20 albo zapisanych bez flagi).
UNVERSIONED = "unversioned"

# Pola tożsamości przebiegu — w kwicie tak, w hashu przepisu nie.
_RUN_IDENTITY_FIELDS = frozenset({"git_sha", "run_id"})


@dataclass(frozen=True)
class PredictionRecipe:
    """Składniki przepisu jednej predykcji.

    Pola:
        llm_provider, llm_model: główny LLM analizy (przy routingu modeli
            provider "routing" i para „tani|frontier").
        council_provider, council_model: LLM rady doradczej.
        prompt_sha256: hash SZABLONU promptu (kod modułów promptów), nie
            wyrenderowanego promptu — ten jest inny dla każdego symbolu.
        personas_sha256: hash zestawu plików person rady.
        config_hash: hash niesekretnych pól konfiguracji (progi, flagi, modele).
        model_version: "cold_start" albo sha256 pliku wag XGBoost.
        git_sha, run_id: commit i przebieg (audyt, poza `version`).
    """

    llm_provider: str
    llm_model: str
    council_provider: str
    council_model: str
    prompt_sha256: str
    personas_sha256: str
    config_hash: str
    model_version: str
    git_sha: str
    run_id: str

    @cached_property
    def version(self) -> str:
        """sha256 kanonicznego JSON-a składników przepisu bez tożsamości przebiegu."""
        recipe = {
            name: value
            for name, value in asdict(self).items()
            if name not in _RUN_IDENTITY_FIELDS
        }
        canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def receipt_fields(self) -> dict[str, str]:
        """Płaskie pola przepisu do kwitu decyzyjnego, razem z `recipe_version`."""
        return {**asdict(self), "recipe_version": self.version}


RECIPE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(PredictionRecipe))


def hit_rate_by_version(
    calls: Iterable[tuple[str | None, str, bool]],
) -> dict[str, DirectionalHitRate]:
    """Trafność kierunkowa per wersja z trójek `(wersja, predicted_trend, is_correct)`.

    Wersja None trafia do segmentu `UNVERSIONED`. Kolejność segmentów = kolejność
    pierwszego wystąpienia (wołający podaje wiersze w swojej kolejności czasu).
    Definicja trafności ta sama co w całym track recordzie (`directional_hit_rate`).
    """
    grouped: dict[str, list[tuple[str, bool]]] = {}
    for version, trend, is_correct in calls:
        grouped.setdefault(version or UNVERSIONED, []).append((trend, is_correct))
    return {version: directional_hit_rate(segment) for version, segment in grouped.items()}
