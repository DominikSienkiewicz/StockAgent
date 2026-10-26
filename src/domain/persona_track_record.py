"""Domena rankingu wiarygodności person rady doradczej (R-3).

Sekcja "historia głosów" pokazuje, KTO jak głosował. Ta domena odpowiada na
pytanie, KTO MIAŁ RACJĘ: zamienia surowe statystyki trafności (hit-rate +
liczba rozliczonych głosów) w uporządkowany, audytowalny leaderboard person.

Czysta logika, ZERO importów zewnętrznych (stdlib + domena) — wejściem są typy
prymitywne, nie DTO warstwy aplikacji. Mapowanie z portu repozytorium na ten
kontrakt robi warstwa wyżej.

Kluczową regułą jest próg `min_votes`. Bez niego persona z jednym trafionym
głosem miałaby 100% trafności i stała na czele rankingu — leaderboard mierzyłby
szum, nie wiarygodność. Próg mieszka TU (a nie w SQL czy w rendererze), bo to
reguła decyzyjna: "poniżej N rozliczonych głosów nie mamy zdania o personie".
Symetrycznie: `PersonaTrackRecord` zawsze niesie `votes`, żeby raport mógł
pokazać wielkość próbki obok procentu — "68% (22 głosy)", nie samo "68%".

WIA-30: `rank_by="wilson"` sortuje po dolnej granicy 95% przedziału Wilsona —
5/5 nie wyprzedza 40/50. WIA-22: `assess_persona_significance` dokłada werdykt
istotności: test dwumianowy na n_eff względem najlepszego naiwnego baseline'u
i korekta BH-FDR po personach (7 person = 7 porównań naraz).

WIA-44: hit-rate ignoruje deklarowaną pewność głosu, więc nagradza przepewne
persony („BUY 0.95” i pudło kosztuje tyle co „BUY 0.55”). `PersonaScoring`
(RPC `persona_scoring_stats`, migracja 065) niesie też średni Brier i log loss
pewności, a `rank_by="brier_skill"` układa leaderboard po Brier Skill względem
stałej prognozy persony (`PersonaTrackRecord.brier_skill`).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from src.domain.council import Recommendation, parse_recommendation
from src.domain.number_format import DecimalMark
from src.domain.prediction import (
    PREDICTION_HORIZON_HOURS,
    Prediction,
    TrendDirection,
    sideways_tolerance_for,
)
from src.domain.significance import (
    DEFAULT_FDR_Q,
    benjamini_hochberg,
    binomial_test_greater,
    verdict_pl,
)
from src.domain.statistics import wilson_interval
from src.domain.track_record import HitRateEvidence
from src.domain.value_objects import AssetType

# Minimalna próbka, przy której hit-rate persony przestaje być szumem.
DEFAULT_MIN_VOTES = 5

RankBy = Literal["hit_rate", "wilson", "brier_skill"]


@dataclass(frozen=True)
class PersonaTrackRecord:
    """Track record jednej persony rady w oknie czasowym.

    - `investor_name` — nazwa persony (np. "Buffett"),
    - `hit_rate` — odsetek trafionych rekomendacji ∈ [0, 1],
    - `votes` — liczba ROZLICZONYCH głosów, na których policzono `hit_rate`,
    - `n_eff` — efektywna liczność głosów (WIA-22); None = nieznana,
    - `p_value` / `significant` — test dwumianowy względem naiwnego baseline'u
      i decyzja po korekcie BH-FDR; None = persona nieoceniona.
    """

    investor_name: str
    hit_rate: float
    votes: int
    n_eff: float | None = None
    p_value: float | None = None
    significant: bool | None = None
    brier: float | None = None
    log_loss: float | None = None

    @property
    def brier_skill(self) -> float | None:
        """WIA-44 — Brier Skill Score względem stałej prognozy persony: 1 − Brier /
        (h·(1 − h)), gdzie h to jej hit-rate. 0 = pewność nie wnosi nic ponad
        „zawsze h”, wartość ujemna = gorzej niż stała prognoza (przepewność).
        None bez Briera albo gdy stała prognoza ma Brier 0 (h = 0 lub 1)."""
        if self.brier is None:
            return None
        climatology = self.hit_rate * (1.0 - self.hit_rate)
        if climatology <= 0.0:
            return None
        return 1.0 - self.brier / climatology

    @property
    def hit_rate_pct(self) -> int:
        """Hit-rate jako pełny procent — gotowy do renderu ("68%")."""
        return round(self.hit_rate * 100)

    @property
    def sample_size(self) -> float:
        """Liczność do przedziału: n_eff, gdy znana, inaczej liczba głosów."""
        return float(self.votes) if self.n_eff is None else self.n_eff

    @property
    def wilson_interval(self) -> tuple[float, float]:
        """95% przedział Wilsona trafności na `sample_size`."""
        return wilson_interval(self.hit_rate * self.sample_size, self.sample_size)

    @property
    def wilson_lower(self) -> float:
        """Dolna granica przedziału — klucz rankingu `rank_by="wilson"` (WIA-30)."""
        return self.wilson_interval[0]

    @property
    def verdict(self) -> str | None:
        """Polski werdykt istotności (decyzja po BH-FDR) z kropką albo None bez oceny."""
        return self.verdict_in(DecimalMark.POINT)

    def verdict_in(self, mark: DecimalMark) -> str | None:
        """Werdykt z p-wartością zapisaną znakiem `mark` (mail: przecinek, MAIL-58)."""
        if self.p_value is None:
            return None
        return verdict_pl(
            self.p_value, self.votes, self.n_eff, mark=mark, significant=self.significant
        )


def sort_personas(
    records: Sequence[PersonaTrackRecord], rank_by: RankBy = "hit_rate"
) -> list[PersonaTrackRecord]:
    """Deterministyczny ranking: malejąco po kluczu `rank_by`, przy remisie
    malejąco po `votes`, a przy remisie obu — alfabetycznie po nazwie.

    `hit_rate` — goły odsetek (dotychczasowe zachowanie); `wilson` (WIA-30) —
    dolna granica 95% przedziału Wilsona, więc mała idealna próba nie wyprzedza
    dużej, mocnej.
    """
    if rank_by == "brier_skill":
        return sorted(
            records,
            key=lambda r: (
                -_skill_or_neutral(r),
                -r.wilson_lower,
                -r.votes,
                r.investor_name,
            ),
        )
    if rank_by == "wilson":
        return sorted(records, key=lambda r: (-r.wilson_lower, -r.votes, r.investor_name))
    return sorted(records, key=lambda r: (-r.hit_rate, -r.votes, r.investor_name))


def _skill_or_neutral(record: PersonaTrackRecord) -> float:
    """Klucz rankingu `brier_skill`: persona bez określonej umiejętności (stała
    prognoza nie ma czego przegrać, np. 10/10) ma 0 — „bez przewagi”, nie ostatnie
    miejsce. Dalej rozstrzyga dolna granica Wilsona."""
    skill = record.brier_skill
    return 0.0 if skill is None else skill


def rank_personas(
    stats: Mapping[str, tuple[float, int]],
    min_votes: int = DEFAULT_MIN_VOTES,
    *,
    rank_by: RankBy = "hit_rate",
) -> list[PersonaTrackRecord]:
    """Buduje leaderboard person z `{investor_name: (hit_rate, votes)}`.

    Persony z próbką mniejszą niż `min_votes` (oraz każda z próbką <= 0, nawet
    przy `min_votes=0`) są odcinane — nie mają wiarygodnego hit-rate'u.
    `hit_rate` jest defensywnie przycinany do [0, 1].

    Sortowanie jest w pełni deterministyczne: malejąco po `hit_rate` (albo po
    dolnej granicy Wilsona przy `rank_by="wilson"`, WIA-30), przy remisie
    malejąco po `votes` (większa próbka = mocniejszy dowód), a przy remisie
    obu — alfabetycznie po nazwie (`sort_personas`). Brak danych → pusta
    lista, co dla warstwy prezentacji oznacza "nie renderuj sekcji".
    """
    records = [
        PersonaTrackRecord(
            investor_name=name,
            hit_rate=min(1.0, max(0.0, hit_rate)),
            votes=votes,
        )
        for name, (hit_rate, votes) in stats.items()
        if votes > 0 and votes >= min_votes
    ]
    return sort_personas(records, rank_by)


@dataclass(frozen=True)
class PersonaScoring:
    """WIA-44 — ocena persony z RPC `persona_scoring_stats` (migracja 065): hit-rate
    i liczba głosów jak w 018 oraz średni Brier (confidence − trafienie)² i log loss
    deklarowanej pewności głosu."""

    hit_rate: float
    votes: int
    brier: float
    log_loss: float


def rank_scored_personas(
    scoring: Mapping[str, PersonaScoring],
    min_votes: int = DEFAULT_MIN_VOTES,
    *,
    rank_by: RankBy = "brier_skill",
) -> list[PersonaTrackRecord]:
    """WIA-44 — leaderboard z oceną Briera z `{investor_name: PersonaScoring}`.

    Próg `min_votes`, przycinanie hit-rate'u i determinizm jak w `rank_personas`;
    domyślnie kolejność po Brier Skill względem stałej prognozy persony, więc
    przepewna persona (wysokie „BUY 0.95” przy tym samym hit-rate) spada pod
    skalibrowaną. Brak danych → pusta lista."""
    records = [
        PersonaTrackRecord(
            investor_name=name,
            hit_rate=min(1.0, max(0.0, score.hit_rate)),
            votes=score.votes,
            brier=score.brier,
            log_loss=score.log_loss,
        )
        for name, score in scoring.items()
        if score.votes > 0 and score.votes >= min_votes
    ]
    return sort_personas(records, rank_by)


def assess_persona_significance(
    records: Sequence[PersonaTrackRecord],
    *,
    baseline: float,
    design_effect: float = 1.0,
    q: float = DEFAULT_FDR_Q,
) -> list[PersonaTrackRecord]:
    """WIA-22 — werdykt istotności każdej persony, kolejność wejścia zachowana.

    `n_eff = votes / design_effect` (efekt projektu ≥ 1: głosy jednego cyklu są
    skorelowane jak predykcje), test dwumianowy jednostronny na n_eff
    zaokrąglonym do całości względem `baseline` (p0 = najlepszy naiwny
    baseline), a decyzja — procedurą BH-FDR na poziomie `q` po wszystkich
    personach naraz. Persona ze świetnym, ale pojedynczym wynikiem wśród
    siedmiu bywa po korekcie „nieodróżnialna od rzutu monetą” — to cena
    uczciwości przy wielu porównaniach.
    """
    effect = max(1.0, design_effect)
    p_values: list[float] = []
    sized: list[PersonaTrackRecord] = []
    for record in records:
        n_eff = record.votes / effect
        trials = max(1, round(n_eff))
        p_values.append(binomial_test_greater(round(record.hit_rate * trials), trials, baseline))
        sized.append(dataclasses.replace(record, n_eff=n_eff))
    rejected = benjamini_hochberg(p_values, q)
    return [
        dataclasses.replace(record, p_value=p, significant=decision)
        for record, p, decision in zip(sized, p_values, rejected, strict=True)
    ]


def assess_against_track_record(
    records: Sequence[PersonaTrackRecord],
    evidence: HitRateEvidence,
    *,
    q: float = DEFAULT_FDR_Q,
) -> list[PersonaTrackRecord]:
    """WIA-22 — ocena person na parametrach z okna scorecardu (bez migracji).

    RPC person zwraca tylko hit-rate i liczbę głosów, więc oba parametry testu
    są PRZYBLIŻENIEM z predykcji okna track recordu: p0 = najlepsza z naiwnych
    strategii „zawsze BULLISH / BEARISH / SIDEWAYS” (`best_any` — persona
    głosuje KUP / SPRZEDAJ / TRZYMAJ), a efekt projektu = n / n_eff predykcji
    kierunkowych. Bez baseline'u albo bez predykcji kierunkowych → rekordy
    bez zmian.
    """
    baseline = evidence.baselines.best_any
    directional = evidence.hit_rate.directional
    if baseline is None or directional == 0 or evidence.n_eff <= 0:
        return list(records)
    return assess_persona_significance(
        records, baseline=baseline[1], design_effect=directional / evidence.n_eff, q=q
    )


# ---------------------------------------------------------------------------
# WIA-51 (G45): reguła trafności głosu persony w domenie
# ---------------------------------------------------------------------------

# Waga konsensusu jak w RPC 015: 0.5 + trafność, przycięta do [0.5, 1.5].
MIN_PERSONA_WEIGHT = 0.5
MAX_PERSONA_WEIGHT = 1.5

# Rekomendacja persony jako kierunek predykcji — ta sama reguła co `reflect`.
_TREND_OF: dict[Recommendation, TrendDirection] = {
    "BUY": TrendDirection.BULLISH,
    "SELL": TrendDirection.BEARISH,
    "HOLD": TrendDirection.SIDEWAYS,
}


@dataclass(frozen=True)
class PersonaVoteOutcome:
    """WIA-51 — jeden rozliczony głos persony: surowy wiersz RPC
    `persona_vote_outcomes` (migracja 045), jeszcze bez oceny.

    Ocena należy do domeny (`persona_vote_correct`), bo SQL nie zna klasy
    aktywa — `symbol` pozwala wybrać pasmo HOLD krypto. `predicted_at`
    i `settled_at` niosą oś czasu dla wag prequential (WIA-46)."""

    prediction_id: str
    investor_name: str
    recommendation: str
    confidence: float
    symbol: str
    price_at_prediction: Decimal
    actual_price: Decimal
    predicted_at: datetime | None
    settled_at: datetime | None

    def is_correct(self, crypto_symbols: Collection[str] = ()) -> bool:
        """Trafność głosu z pasmem HOLD klasy aktywa jego symbolu."""
        asset_type = AssetType.CRYPTO if self.symbol.upper() in crypto_symbols else None
        return persona_vote_correct(
            self.recommendation, self.price_at_prediction, self.actual_price, asset_type
        )


def persona_vote_correct(
    recommendation: str,
    price_at_prediction: Decimal,
    actual_price: Decimal,
    asset_type: AssetType | None = None,
) -> bool:
    """WIA-51 — czy rekomendacja persony potwierdziła się na rozliczonej cenie.

    BUY = wzrost, SELL = spadek, HOLD = ruch w paśmie „bez ruchu” klasy aktywa
    (`sideways_tolerance_for`: krypto ±2.5%, reszta ±0.5%) — dokładnie reguła
    `Prediction.is_trend_correct`, którą `reflect` ocenia predykcję. SQL z 015
    i 018 oceniał HOLD na krypto pasmem akcji, czyli 5× ostrzej niż domena.
    Rekomendacja spoza BUY/SELL/HOLD to głos chybiony (jak `ELSE 0.0` w SQL)."""
    parsed = parse_recommendation(recommendation)
    if parsed is None:
        return False
    call = Prediction(
        symbol="",
        predicted_trend=_TREND_OF[parsed],
        price_at_prediction=price_at_prediction,
        predicted_target_price=price_at_prediction,
    )
    return call.is_trend_correct(actual_price, tolerance=sideways_tolerance_for(asset_type))


def persona_hit_stats(
    outcomes: Iterable[PersonaVoteOutcome], crypto_symbols: Collection[str] = ()
) -> dict[str, tuple[float, int]]:
    """`{investor_name: (hit_rate, liczba_głosów)}` — kontrakt RPC 018 liczony
    regułą domeny; wejście `rank_personas`."""
    votes: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for outcome in outcomes:
        votes[outcome.investor_name] += 1
        hits[outcome.investor_name] += outcome.is_correct(crypto_symbols)
    return {name: (hits[name] / count, count) for name, count in votes.items()}


def persona_weight(hit_rate: float) -> float:
    """Waga persony w konsensusie: `0.5 + hit_rate` w [0.5, 1.5] (jak RPC 015)."""
    return max(MIN_PERSONA_WEIGHT, min(MAX_PERSONA_WEIGHT, MIN_PERSONA_WEIGHT + hit_rate))


def persona_accuracy_weights(
    outcomes: Iterable[PersonaVoteOutcome], crypto_symbols: Collection[str] = ()
) -> dict[str, float]:
    """`{investor_name: waga}` — kontrakt RPC 015 na tej samej regule co
    `persona_hit_stats`, więc wagi i leaderboard nie mogą się rozjechać."""
    return {
        name: persona_weight(hit_rate)
        for name, (hit_rate, _) in persona_hit_stats(outcomes, crypto_symbols).items()
    }


# ---------------------------------------------------------------------------
# WIA-46: wagi prequential — tylko głosy rozliczone przed chwilą predykcji
# ---------------------------------------------------------------------------


def outcome_known_at(outcome: PersonaVoteOutcome) -> datetime | None:
    """Najwcześniejsza chwila, w której wynik głosu był znany.

    `settled_at` (trigger 029), gdy jest późniejszy niż predykcja. Backfill 029
    zostawia dolną granicę `settled_at = timestamp`, a wiersz bez stempla nie ma
    nic — wtedy przybliżamy chwilą predykcji plus horyzont prognozy
    (`PREDICTION_HORIZON_HOURS`): wcześniej wyniku nie było. Bez chwili
    predykcji głosu nie da się umieścić na osi czasu → None."""
    if outcome.predicted_at is None:
        return None
    if outcome.settled_at is not None and outcome.settled_at > outcome.predicted_at:
        return outcome.settled_at
    return outcome.predicted_at + timedelta(hours=PREDICTION_HORIZON_HOURS)


def is_known_before(outcome: PersonaVoteOutcome, as_of: datetime, window_days: int) -> bool:
    """Czy głos należy do wag z chwili `as_of`: predykcja z okna `window_days`
    przed `as_of` i wynik znany ŚCIŚLE przed `as_of`.

    Okno liczy się po chwili predykcji, jak `pl.timestamp >= NOW() - okno`
    w RPC 015, tylko od `as_of` zamiast od dziś."""
    known_at = outcome_known_at(outcome)
    if known_at is None or outcome.predicted_at is None:
        return False
    return outcome.predicted_at >= as_of - timedelta(days=window_days) and known_at < as_of


def outcomes_known_before(
    outcomes: Iterable[PersonaVoteOutcome], as_of: datetime, window_days: int
) -> list[PersonaVoteOutcome]:
    """Głosy, które Fast Loop znał w chwili `as_of` (`is_known_before`)."""
    return [outcome for outcome in outcomes if is_known_before(outcome, as_of, window_days)]


def persona_accuracy_weights_as_of(
    outcomes: Iterable[PersonaVoteOutcome],
    *,
    as_of: datetime,
    window_days: int = 90,
    crypto_symbols: Collection[str] = (),
) -> dict[str, float]:
    """WIA-46 — wagi person, które Fast Loop MÓGŁ znać w chwili `as_of`.

    RPC 015 liczy okno od NOW(), więc retrospektywne „czy ważenie pomaga”
    użyłoby wag znających przyszłość i zawyżyło wynik."""
    known = outcomes_known_before(outcomes, as_of, window_days)
    return persona_accuracy_weights(known, crypto_symbols)
