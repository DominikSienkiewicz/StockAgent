"""Champion/challenger nogi ML (ML-33): reguła promocji na predykcjach live.

Shipowany model jest refitowany na całości i nie jest re-walidowany (METRIC_NOTE
adaptera XGBoost), więc jedyna uczciwa ocena poza próbą to predykcje live. Fast Loop
zapisuje prognozę championa (tę, która poszła do maila) i prognozy cienia
challengerów do `shadow_predictions`, a Slow Loop porównuje je PARAMI — na tych
samych predykcjach, po rozliczeniu.

Reguła jest pre-rejestrowana: to bramka walk-forward zastosowana do danych live, z
tymi samymi progami (`MIN_REL_IMPROVEMENT`, `MIN_OOS_POINTS`, `PAIRED_Z_CRIT`),
zapisana w kodzie, zanim zebrały się wyniki. Werdykt trafia do summary Slow Loopa;
podmiana serwowanych wag zostaje decyzją właściciela, nie automatem.

Straty pary:
- challenger zwrotu (regresor) — kwadrat błędu zwrotu 12 h, jak RMSE bramki;
- challenger kierunku (klasyfikator ML-30) — Brier zdarzenia „wzrost poza martwą
  strefą” (etykieta `direction_label`, wiersze w martwej strefie wypadają); champion
  dostaje twarde p = 1 / 0 ze znaku swojej prognozy (0,5 przy zerze), bo kierunek
  to wszystko, co jego liczba mówi o tym zdarzeniu.

Czysta domena, stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from src.domain.backtest import (
    MIN_OOS_POINTS,
    MIN_REL_IMPROVEMENT,
    PAIRED_Z_CRIT,
    paired_loss_z,
)
from src.domain.direction_variant import direction_label

# `model_name` wiersza championa w `shadow_predictions`; challengerzy mają nazwy
# z `ml_challengers` w config.toml.
CHAMPION_MODEL_NAME = "champion"

# Pre-rejestracja: progi bramki walk-forward, nie strojone pod wyniki live.
MIN_LIVE_N = MIN_OOS_POINTS
PROMOTION_MARGIN = MIN_REL_IMPROVEMENT
PROMOTION_Z_CRIT = PAIRED_Z_CRIT
# Okno rozliczonych predykcji, z którego Slow Loop liczy werdykt.
LIVE_WINDOW_DAYS = 90


class ForecastKind(StrEnum):
    """Co mówi prognoza modelu: zwrot 12 h (regresor) albo P(wzrost) (klasyfikator)."""

    RETURN = "return"
    DIRECTION = "direction"


class PromotionStatus(StrEnum):
    PROMOTE = "promote"
    KEEP_CHAMPION = "keep_champion"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class ShadowForecast:
    """Prognoza jednego modelu nogi ML dla jednej predykcji.

    `model_version` to sha256 pliku wag (jak `model_version` przepisu, CI-13).
    `predicted_return` — zwrot 12 h w surowych jednostkach (model w jednostkach σ
    jest już przemnożony przez σ), `proba` — P(zwrot > +martwa strefa)."""

    model_name: str
    model_version: str
    predicted_return: float | None = None
    proba: float | None = None

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_version:
            raise ValueError("shadow forecast needs a model name and a model version")
        if self.predicted_return is None and self.proba is None:
            raise ValueError("shadow forecast needs a predicted return or a probability")
        if self.predicted_return is not None and not math.isfinite(self.predicted_return):
            raise ValueError(f"predicted return must be finite, got {self.predicted_return}")
        if self.proba is not None and not 0.0 <= self.proba <= 1.0:
            raise ValueError(f"probability must lie in [0, 1], got {self.proba}")

    @property
    def kind(self) -> ForecastKind:
        return ForecastKind.RETURN if self.predicted_return is not None else ForecastKind.DIRECTION

    @property
    def is_champion(self) -> bool:
        return self.model_name == CHAMPION_MODEL_NAME

    @property
    def up_probability(self) -> float:
        """P(wzrost): klasyfikator podaje ją wprost, prognoza zwrotu to twardy kierunek."""
        if self.proba is not None:
            return self.proba
        predicted = self.predicted_return or 0.0
        return 0.5 if predicted == 0 else (1.0 if predicted > 0 else 0.0)


@dataclass(frozen=True)
class SettledShadowForecast:
    """Prognoza cienia z rozliczonej predykcji — `realized_return` to zwrot 12 h
    z `prediction_logs` (cena rozliczenia względem ceny przy predykcji)."""

    prediction_id: str
    forecast: ShadowForecast
    realized_return: float


@dataclass(frozen=True)
class PromotionVerdict:
    """Werdykt pary champion–challenger na tych samych predykcjach live.

    Straty to średnie po parach; `paired_z` — statystyka testu sparowanego
    (EXP-04, d = strata championa − strata challengera, dodatnie = challenger lepszy)."""

    status: PromotionStatus
    n_pairs: int
    champion_loss: float | None
    challenger_loss: float | None
    paired_z: float | None

    @property
    def promote(self) -> bool:
        return self.status is PromotionStatus.PROMOTE

    @property
    def improvement(self) -> float | None:
        """Względny spadek straty challengera wobec championa (None bez strat albo
        przy bezbłędnym championie)."""
        if not self.champion_loss or self.challenger_loss is None:
            return None
        return (self.champion_loss - self.challenger_loss) / self.champion_loss


@dataclass(frozen=True)
class ChallengerVerdict:
    """Werdykt jednego challengera; `n_versions` — ile wersji jego wag dało pary w oknie."""

    model_name: str
    kind: ForecastKind
    n_versions: int
    verdict: PromotionVerdict


def should_promote(
    champion: Sequence[float],
    challenger: Sequence[float],
    *,
    min_live_n: int = MIN_LIVE_N,
    margin: float = PROMOTION_MARGIN,
    z_crit: float = PROMOTION_Z_CRIT,
) -> PromotionVerdict:
    """Czy challenger powinien zastąpić championa — na stratach SPAROWANYCH
    (i-ta strata obu modeli dotyczy tej samej predykcji live).

    Promocja wymaga co najmniej `min_live_n` par, średniej straty challengera niższej
    o co najmniej `margin` (względnie) i testu sparowanego z > `z_crit`. Bezbłędnego
    championa nie da się pobić. Poniżej `min_live_n` werdykt to brak danych."""
    if len(champion) != len(challenger):
        raise ValueError("champion and challenger losses must be paired one to one")
    if min_live_n < 2:
        raise ValueError(f"min_live_n must be at least 2 for the paired test, got {min_live_n}")
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must lie in [0, 1), got {margin}")
    n = len(champion)
    if n == 0:
        return PromotionVerdict(PromotionStatus.INSUFFICIENT_DATA, 0, None, None, None)
    champion_loss = math.fsum(champion) / n
    challenger_loss = math.fsum(challenger) / n
    diffs = [ours - theirs for ours, theirs in zip(champion, challenger, strict=True)]
    z = paired_loss_z(
        [
            {
                "n_valid": float(n),
                "sum_loss_diff": math.fsum(diffs),
                "sum_loss_diff_sq": math.fsum(d * d for d in diffs),
            }
        ]
    )
    if n < min_live_n:
        status = PromotionStatus.INSUFFICIENT_DATA
    elif (
        champion_loss > 0
        and challenger_loss <= champion_loss * (1.0 - margin)
        and z is not None
        and z > z_crit
    ):
        status = PromotionStatus.PROMOTE
    else:
        status = PromotionStatus.KEEP_CHAMPION
    return PromotionVerdict(status, n, champion_loss, challenger_loss, z)


def paired_loss(
    champion: ShadowForecast,
    challenger: ShadowForecast,
    realized_return: float,
    *,
    dead_band: float,
) -> tuple[float, float] | None:
    """(strata championa, strata challengera) jednej rozliczonej predykcji albo None,
    gdy para nie ma wspólnej miary (nieznany wynik, ruch w martwej strefie przy
    challengerze kierunku, champion bez prognozy zwrotu przy regresorze)."""
    if not math.isfinite(realized_return):
        return None
    if challenger.kind is ForecastKind.RETURN:
        ours, theirs = champion.predicted_return, challenger.predicted_return
        if ours is None or theirs is None:
            return None
        return (realized_return - ours) ** 2, (realized_return - theirs) ** 2
    outcome = direction_label(realized_return, dead_band=dead_band)
    if outcome is None:
        return None
    return (
        (champion.up_probability - outcome) ** 2,
        (challenger.up_probability - outcome) ** 2,
    )


def challenger_verdicts(
    rows: Sequence[SettledShadowForecast],
    *,
    dead_band: float,
    min_live_n: int = MIN_LIVE_N,
    margin: float = PROMOTION_MARGIN,
    z_crit: float = PROMOTION_Z_CRIT,
) -> tuple[ChallengerVerdict, ...]:
    """Werdykt każdego challengera (nazwa i rodzaj prognozy) wobec championa tej
    samej predykcji. Challenger bez wiersza championa (cold start) nie ma pary.
    Kolejność: po nazwie, potem po rodzaju."""
    champions = {row.prediction_id: row.forecast for row in rows if row.forecast.is_champion}
    groups: dict[tuple[str, ForecastKind], list[SettledShadowForecast]] = {}
    for row in rows:
        if not row.forecast.is_champion:
            groups.setdefault((row.forecast.model_name, row.forecast.kind), []).append(row)
    verdicts: list[ChallengerVerdict] = []
    for (name, kind), members in sorted(groups.items()):
        ours: list[float] = []
        theirs: list[float] = []
        for row in members:
            champion = champions.get(row.prediction_id)
            losses = (
                None
                if champion is None
                else paired_loss(champion, row.forecast, row.realized_return, dead_band=dead_band)
            )
            if losses is not None:
                ours.append(losses[0])
                theirs.append(losses[1])
        verdicts.append(
            ChallengerVerdict(
                model_name=name,
                kind=kind,
                n_versions=len({row.forecast.model_version for row in members}),
                verdict=should_promote(
                    ours, theirs, min_live_n=min_live_n, margin=margin, z_crit=z_crit
                ),
            )
        )
    return tuple(verdicts)
