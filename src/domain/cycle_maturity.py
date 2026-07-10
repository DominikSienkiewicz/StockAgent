"""Domena: dojrzałość cyklu — odróżnienie "Dnia 1" od stanu ustalonego.

Pierwszy cykl nowego deploymentu nie ma punktu odniesienia (poprzedniej ceny),
więc wszystkie symbole trafiają do "⏸ Pominięte" jako cold-start. Bez tej
klasyfikacji taki mail jest nieodróżnialny od cyklu, w którym padły wszystkie
źródła cen — a to dwie zupełnie różne historie: "przemyślany produkt buduje
punkt odniesienia" kontra "coś jest zepsute".

Reguła żyje w domenie (czysty stdlib, bez I/O). Graf/orkiestrator jest tylko
wykonawcą: zlicza wyniki cyklu na prymitywy i przekazuje je tutaj.

Największe ryzyko, które ta reguła musi wykluczyć: pomylenie cold-startu
z masową awarią źródeł. Dlatego przyczyny pominięcia są rozdzielone
(`SkipReason`), a błędy liczone jawnie osobno — powitalny ton nigdy nie może
przykryć incydentu.
"""

from __future__ import annotations

from enum import Enum


class SkipReason(Enum):
    """Powód, dla którego symbol nie dostał świeżej predykcji w tym cyklu.

    Orkiestrator wystawi tę wartość jako pole na DTO application (per symbol),
    żeby renderer mógł rozdzielić przyczyny pominięcia zamiast wrzucać wszystko
    do jednego worka "⏸ Pominięte".

    - `COLD_START` — brak poprzedniej ceny (pierwszy cykl / nowy symbol); brak
      punktu odniesienia dla bramki volatility. To NIE jest awaria.
    - `BELOW_THRESHOLD` — poprzednia cena istniała, ale zmiana < progu
      volatility; bramka FinOps świadomie wstrzymała płatne wywołania.
    - `UNSUPPORTED_PRICE` — symbol nieobsługiwany przez bieżący adapter cen
      (np. EU-listed na Finnhub free); pre-filtrowany wyżej, wykluczony
      z mianownika cold-startu.
    """

    COLD_START = "COLD_START"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    UNSUPPORTED_PRICE = "UNSUPPORTED_PRICE"


class CycleMaturity(Enum):
    """Faza życia cyklu z perspektywy powitania użytkownika.

    - `FIRST_RUN` — "Dzień 1": budujemy punkt odniesienia, pierwsze predykcje
      dopiero jutro. Uzasadnia powitalny ton i temat maila.
    - `STEADY_STATE` — normalny cykl (są predykcje, są punkty odniesienia)
      LUB sytuacja, której NIE wolno ubrać w powitanie (masowa awaria, cykl
      pusty). Ścieżka domyślna raportu.
    """

    FIRST_RUN = "FIRST_RUN"
    STEADY_STATE = "STEADY_STATE"


def classify_cycle(
    *,
    saved_count: int,
    cold_start_count: int,
    below_threshold_count: int,
    error_count: int,
    unsupported_count: int = 0,
) -> CycleMaturity:
    """Klasyfikuje dojrzałość cyklu na podstawie zliczeń jego wyników.

    Wszystkie argumenty to prymitywy (liczności) wyliczone przez orkiestratora
    z wyników cyklu — domena nie zna DTO application (`SymbolResult`).

    Mianownik "obsługiwalności" to symbole, które realnie mogły wygenerować
    predykcję: `saved + cold_start + below_threshold + error`. Symbole
    `unsupported_count` są z niego JAWNIE wykluczone (trzecia kategoria
    "ignored", pre-filtrowana wyżej) — nie mogą ani zrobić z incydentu Dnia 1,
    ani odwrotnie.

    Cykl jest `FIRST_RUN` tylko gdy jest to prawdziwy pierwszy kontakt:

    - nic jeszcze nie zapisano (`saved_count == 0` → nie istnieje historia),
    - nic nie było "poniżej progu" (`below_threshold_count == 0` → gdyby było,
      istniałaby poprzednia cena, więc punkt odniesienia już był),
    - są realne cold-starty (`cold_start_count > 0`),
    - cold-start PRZEWAŻA nad błędami (`cold_start_count > error_count`) — to
      rozdzielenie zamraża największe ryzyko: cykl zdominowany przez awarie
      źródeł nigdy nie zostanie sklasyfikowany jako powitalny Dzień 1.

    W każdym innym przypadku → `STEADY_STATE` (normalny raport / incydent /
    cykl pusty), którego ton nie udaje powitania.
    """
    has_history = saved_count > 0 or below_threshold_count > 0
    cold_start_dominates = cold_start_count > 0 and cold_start_count > error_count

    if not has_history and cold_start_dominates:
        return CycleMaturity.FIRST_RUN
    return CycleMaturity.STEADY_STATE
