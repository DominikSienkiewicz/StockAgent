"""Domena rankingu wiarygodności person rady doradczej (#3).

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
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.domain.number_format import DecimalMark
from src.domain.significance import (
    DEFAULT_FDR_Q,
    benjamini_hochberg,
    binomial_test_greater,
    verdict_pl,
)
from src.domain.statistics import wilson_interval
from src.domain.track_record import HitRateEvidence

# Minimalna próbka, przy której hit-rate persony przestaje być szumem.
DEFAULT_MIN_VOTES = 5

RankBy = Literal["hit_rate", "wilson"]


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
    if rank_by == "wilson":
        return sorted(records, key=lambda r: (-r.wilson_lower, -r.votes, r.investor_name))
    return sorted(records, key=lambda r: (-r.hit_rate, -r.votes, r.investor_name))


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
