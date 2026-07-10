"""Domena rankingu wiarygodności person rady doradczej (#3).

Sekcja "historia głosów" pokazuje, KTO jak głosował. Ta domena odpowiada na
pytanie, KTO MIAŁ RACJĘ: zamienia surowe statystyki trafności (hit-rate +
liczba rozliczonych głosów) w uporządkowany, audytowalny leaderboard person.

Czysta logika, ZERO importów zewnętrznych (tylko stdlib) — wejściem są typy
prymitywne, nie DTO warstwy aplikacji. Mapowanie z portu repozytorium na ten
kontrakt robi warstwa wyżej.

Kluczową regułą jest próg `min_votes`. Bez niego persona z jednym trafionym
głosem miałaby 100% trafności i stała na czele rankingu — leaderboard mierzyłby
szum, nie wiarygodność. Próg mieszka TU (a nie w SQL czy w rendererze), bo to
reguła decyzyjna: "poniżej N rozliczonych głosów nie mamy zdania o personie".
Symetrycznie: `PersonaTrackRecord` zawsze niesie `votes`, żeby raport mógł
pokazać wielkość próbki obok procentu — "68% (22 głosy)", nie samo "68%".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Minimalna próbka, przy której hit-rate persony przestaje być szumem.
DEFAULT_MIN_VOTES = 5


@dataclass(frozen=True)
class PersonaTrackRecord:
    """Track record jednej persony rady w oknie czasowym.

    - `investor_name` — nazwa persony (np. "Buffett"),
    - `hit_rate` — odsetek trafionych rekomendacji ∈ [0, 1],
    - `votes` — liczba ROZLICZONYCH głosów, na których policzono `hit_rate`.
    """

    investor_name: str
    hit_rate: float
    votes: int

    @property
    def hit_rate_pct(self) -> int:
        """Hit-rate jako pełny procent — gotowy do renderu ("68%")."""
        return round(self.hit_rate * 100)


def rank_personas(
    stats: Mapping[str, tuple[float, int]],
    min_votes: int = DEFAULT_MIN_VOTES,
) -> list[PersonaTrackRecord]:
    """Buduje leaderboard person z `{investor_name: (hit_rate, votes)}`.

    Persony z próbką mniejszą niż `min_votes` (oraz każda z próbką <= 0, nawet
    przy `min_votes=0`) są odcinane — nie mają wiarygodnego hit-rate'u.
    `hit_rate` jest defensywnie przycinany do [0, 1].

    Sortowanie jest w pełni deterministyczne: malejąco po `hit_rate`, przy
    remisie malejąco po `votes` (większa próbka = mocniejszy dowód), a przy
    remisie obu — alfabetycznie po nazwie. Brak danych → pusta lista, co dla
    warstwy prezentacji oznacza "nie renderuj sekcji".
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
    records.sort(key=lambda r: (-r.hit_rate, -r.votes, r.investor_name))
    return records
