"""WIA-43 — wkład krańcowy persony: leave-one-out na zapisanych głosach (offline).

Hit-rate persony (018) nie mówi, czy jej głos KIEDYKOLWIEK zmienia decyzję rady:
persona trafna, ale zawsze zgodna z większością, wnosi zero i kosztuje wywołanie.
Tu dla każdej rozliczonej predykcji z zapisanymi głosami przeliczamy werdykt rady
tą samą regułą co Fast Loop (`derive_consensus`) bez persony X i sprawdzamy, czy
`final_recommendation` by trafiła — regułą głosu persony (`persona_vote_correct`,
pasmo HOLD wg klasy aktywa). Wynik per persona: Δ trafności i Δ Briera siły
konsensusu po jej usunięciu, liczba odwróceń werdyktu i przedziały z bootstrapu
blokowego po cyklach (dniach UTC predykcji).

Czysta domena, zero I/O. Wejściem są surowe głosy z RPC `persona_vote_outcomes`
(migracja 045).

Ograniczenie metody: leave-one-out ignoruje INTERAKCJE. Dwie redundantne persony
(np. Graham i Buffett zgadzający się w 95%) wyjdą OBIE na zero, bo każda z nich
zastępuje drugą — dlatego wynik czyta się razem z redundancją rady (WIA-45,
`council_redundancy`): kandydata do wycięcia usuwa się pojedynczo i mierzy od nowa.
Predykcje z jednym głosem odpadają: rada bez persony X byłaby pusta.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.domain.council import InvestorOpinion, derive_consensus, parse_recommendation
from src.domain.persona_track_record import PersonaVoteOutcome, persona_vote_correct
from src.domain.significance import cluster_bootstrap_statistic_ci
from src.domain.track_record import group_by_cycle
from src.domain.value_objects import AssetType

# Rada z mniejszą liczbą głosów nie ma czego porównywać: bez persony X byłaby pusta.
MIN_COUNCIL_VOTES = 2


@dataclass(frozen=True)
class PersonaMarginal:
    """Wkład krańcowy jednej persony (usunięcie = werdykt rady bez niej)."""

    investor_name: str
    predictions: int
    flips: int
    accuracy_with: float
    accuracy_without: float
    brier_with: float
    brier_without: float
    accuracy_interval: tuple[float, float] | None
    brier_interval: tuple[float, float] | None

    @property
    def accuracy_delta(self) -> float:
        """Zmiana trafności werdyktu po usunięciu persony (ujemna = persona pomaga)."""
        return self.accuracy_without - self.accuracy_with

    @property
    def brier_delta(self) -> float:
        """Zmiana Briera siły konsensusu po usunięciu persony (dodatnia = persona pomaga)."""
        return self.brier_without - self.brier_with

    @property
    def flip_rate(self) -> float:
        return self.flips / self.predictions

    @property
    def removal_candidate(self) -> bool:
        """Usunięcie nie pogarsza ani trafności, ani Briera werdyktu rady (estymaty
        punktowe). To sygnał do przeglądu, nie decyzja: CI może obejmować zero,
        a interakcje (redundantne pary) leave-one-out pomija."""
        return self.accuracy_delta >= 0.0 and self.brier_delta <= 0.0


@dataclass(frozen=True)
class CouncilAblation:
    """Leave-one-out całej rady: `predictions` ocenionych predykcji (≥ 2 głosy),
    trafność i Brier pełnej rady oraz wkład każdej persony."""

    predictions: int
    accuracy: float | None
    brier: float | None
    personas: tuple[PersonaMarginal, ...]


@dataclass(frozen=True)
class _Effect:
    """Skutek usunięcia jednej persony z jednej predykcji."""

    cycle: str
    flipped: bool
    hit_with: bool
    hit_without: bool
    brier_with: float
    brier_without: float


def _opinions(votes: Sequence[PersonaVoteOutcome]) -> list[InvestorOpinion]:
    """Głosy predykcji jako opinie rady; rekomendacja spoza BUY/SELL/HOLD odpada,
    a duplikat persony w jednej predykcji liczy się raz (pierwszy głos)."""
    opinions: dict[str, InvestorOpinion] = {}
    for vote in votes:
        recommendation = parse_recommendation(vote.recommendation)
        if recommendation is not None and vote.investor_name not in opinions:
            opinions[vote.investor_name] = InvestorOpinion(
                vote.investor_name, recommendation, vote.confidence, "", ()
            )
    return list(opinions.values())


def _score(
    opinions: list[InvestorOpinion], sample: PersonaVoteOutcome, asset_type: AssetType | None
) -> tuple[str, bool, float]:
    """(rekomendacja, trafienie, kwadrat błędu siły konsensusu) werdyktu z `opinions`."""
    recommendation, strength = derive_consensus(opinions)
    hit = persona_vote_correct(
        recommendation, sample.price_at_prediction, sample.actual_price, asset_type
    )
    return recommendation, hit, (strength - (1.0 if hit else 0.0)) ** 2


def _cycle_key(moment: datetime) -> str:
    """Dzień UTC predykcji — predykcje jednego dnia są skorelowane (blok bootstrapu)."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).date().isoformat()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _block_interval(
    effects: Sequence[_Effect], delta: str, n_boot: int, seed: int
) -> tuple[float, float] | None:
    """95% przedział średniej różnicy (bez persony − z personą), cykle ze zwracaniem."""
    keyed = [
        (
            e.cycle,
            float(e.hit_without) - float(e.hit_with)
            if delta == "accuracy"
            else e.brier_without - e.brier_with,
        )
        for e in effects
    ]
    return cluster_bootstrap_statistic_ci(
        group_by_cycle(keyed), _mean, n_boot=n_boot, seed=seed
    )


def _marginal(name: str, effects: Sequence[_Effect], n_boot: int, seed: int) -> PersonaMarginal:
    return PersonaMarginal(
        investor_name=name,
        predictions=len(effects),
        flips=sum(1 for e in effects if e.flipped),
        accuracy_with=_mean([float(e.hit_with) for e in effects]),
        accuracy_without=_mean([float(e.hit_without) for e in effects]),
        brier_with=_mean([e.brier_with for e in effects]),
        brier_without=_mean([e.brier_without for e in effects]),
        accuracy_interval=_block_interval(effects, "accuracy", n_boot, seed),
        brier_interval=_block_interval(effects, "brier", n_boot, seed),
    )


def leave_one_out(
    outcomes: Sequence[PersonaVoteOutcome],
    *,
    since: datetime | None = None,
    crypto_symbols: Collection[str] = (),
    n_boot: int = 2000,
    seed: int = 0,
) -> CouncilAblation:
    """Wkład krańcowy każdej persony na rozliczonych predykcjach od `since`
    (None = wszystkie). `crypto_symbols` wybiera pasmo HOLD krypto. Persony
    posortowane: najpierw kandydaci do wycięcia (największe Δ trafności),
    potem alfabetycznie."""
    crypto = frozenset(symbol.upper() for symbol in crypto_symbols)
    by_prediction: dict[str, list[PersonaVoteOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.predicted_at is not None and (since is None or outcome.predicted_at >= since):
            by_prediction[outcome.prediction_id].append(outcome)

    effects: dict[str, list[_Effect]] = defaultdict(list)
    hits: list[float] = []
    briers: list[float] = []
    for votes in by_prediction.values():
        opinions = _opinions(votes)
        sample = votes[0]
        if len(opinions) < MIN_COUNCIL_VOTES or sample.predicted_at is None:
            continue
        asset_type = AssetType.CRYPTO if sample.symbol.upper() in crypto else None
        full_recommendation, hit_with, brier_with = _score(opinions, sample, asset_type)
        hits.append(float(hit_with))
        briers.append(brier_with)
        for opinion in opinions:
            rest = [other for other in opinions if other is not opinion]
            recommendation, hit_without, brier_without = _score(rest, sample, asset_type)
            effects[opinion.investor_name].append(
                _Effect(
                    cycle=_cycle_key(sample.predicted_at),
                    flipped=recommendation != full_recommendation,
                    hit_with=hit_with,
                    hit_without=hit_without,
                    brier_with=brier_with,
                    brier_without=brier_without,
                )
            )

    personas = sorted(
        (_marginal(name, items, n_boot, seed) for name, items in effects.items()),
        key=lambda m: (-m.accuracy_delta, m.brier_delta, m.investor_name),
    )
    return CouncilAblation(
        predictions=len(hits),
        accuracy=_mean(hits) if hits else None,
        brier=_mean(briers) if briers else None,
        personas=tuple(personas),
    )
