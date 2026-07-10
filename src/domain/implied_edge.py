"""Domena: edge modelu vs rynek opcji — dywergencja predykcji od implied move.

Repo ma obie strony równania — przewidywany ruch XGBoosta i implikowaną zmienność
z `OptionsFlowSnapshot` — i dotąd ich nie stykało. Ten moduł je styka: skaluje
annualizowaną IV na horyzont predykcji (sqrt-time) i porównuje z ruchem modelu,
odpowiadając na pytanie „czy model wie coś, czego rynek nie wycenił?".

Czysta logika, ZERO importów zewnętrznych — tylko `math` (float + `math.sqrt`,
dokładnie jak w `correlation.py`). Mieszanie z `Decimal` wywala mypy strict, więc
API operuje wyłącznie na `float`.

Granica Decimal → float
-----------------------
Ceny w reszcie systemu bywają `Decimal` (`predicted_target_price`, cena bieżąca).
Konwersję na `float` wykonuje WOŁAJĄCY na granicy modułu (np. `float(price)` w
`_predict_node`). Ten moduł nigdy nie przyjmuje `Decimal` — to celowe, żeby
utrzymać jednorodny typ i przejść mypy strict.

OGRANICZENIA METODYCZNE (jawne, wiążące)
----------------------------------------
1. `implied_vol` w `OptionsFlowSnapshot` to ŚREDNIA PO CAŁYM ŁAŃCUCHU opcji, a nie
   ATM najbliższej ekspiracji. Implied move liczony z tej średniej jest więc
   przybliżeniem zgrubnym, nie precyzyjną wyceną ruchu at-the-money.
2. Sqrt-time scaling zakłada ciągłość handlu. Zastosowany na godzinach
   KALENDARZOWYCH przez weekend PRZESZACOWUJE ruch (12h kalendarzowych != 12h
   handlu — rynek jest zamknięty). To znany bias w stronę zawyżania implied move.

Dlatego `edge_sigma` jest SYGNAŁEM INFORMACYJNYM do walidacji na zamkniętych
predykcjach — NIE MA (jeszcze) roli decyzyjnej. Zaczynamy od etykiet JAKOŚCIOWYCH
(`EdgeLabel`), nie od surowych liczb σ udających precyzję.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Godziny kalendarzowe w roku — baza annualizacji sqrt-time (patrz ograniczenie #2).
_HOURS_PER_YEAR = 365.0 * 24.0

# Progi klasyfikacji jakościowej na skali edge_sigma = |ruch modelu| / implied move.
# Powyżej: model przewiduje ruch większy, niż rynek wycenia (potencjalny edge).
_MODEL_AHEAD_SIGMA = 1.5
# Poniżej: rynek wycenia ruch, którego model nie widzi (model „śpi").
_MARKET_AHEAD_SIGMA = 0.5


class EdgeLabel(Enum):
    """Jakościowa etykieta dywergencji modelu od rynku opcji.

    Świadomie zaczynamy od etykiet, nie od liczb σ z dwoma miejscami po przecinku —
    surowy `edge_sigma` (przy ograniczeniach metodycznych modułu) nie zasługuje na
    pozorną precyzję.
    """

    MODEL_AHEAD = "MODEL_AHEAD"  # model widzi więcej, niż rynek wycenił
    MARKET_AHEAD = "MARKET_AHEAD"  # rynek wycenia ruch, model śpi
    ALIGNED = "ALIGNED"  # model i rynek zgodni co do skali ruchu
    NO_SIGNAL = "NO_SIGNAL"  # brak IV / dane bez sensu → brak sygnału


@dataclass(frozen=True)
class ImpliedEdge:
    """Wynik porównania ruchu modelu z implikowanym ruchem rynku opcji.

    `label` — etykieta jakościowa (raport pokazuje właśnie ją). `edge_sigma` —
    surowa krotność implied move (None, gdy brak sygnału). `implied_move` — ułamkowy
    ruch implikowany na horyzont (None, gdy brak IV). `model_move` — ułamkowy,
    ZNAKOWANY ruch przewidywany przez model.

    Świadomie NIE niesie żadnego pola akcji/rekomendacji/gate — to sygnał
    informacyjny do walidacji, nie sterownik decyzji.
    """

    label: EdgeLabel
    edge_sigma: float | None
    implied_move: float | None
    model_move: float


def implied_move_12h(iv: float, horizon_hours: float) -> float:
    """Implikowany ułamkowy ruch na `horizon_hours`, sqrt-time z annualizowanej IV.

    `sigma_okres = iv * sqrt(horyzont / rok)` na godzinach kalendarzowych. IV podana
    jako ułamek (0.45 = 45% annualizowane). Zwraca ułamek (0.017 = ±1.7%).

    Bramka: `iv <= 0` lub `horizon_hours <= 0` → 0.0 (brak sensownego ruchu, nigdy
    dzielenia przez zero). Nazwa `_12h` odnosi się do domyślnego zastosowania
    (fast-loop 12h), ale funkcja skaluje dowolny horyzont.

    UWAGA: przez weekend (godziny kalendarzowe) sqrt-time przeszacowuje — patrz
    ograniczenia metodyczne w docstringu modułu.
    """
    if iv <= 0.0 or horizon_hours <= 0.0:
        return 0.0
    return iv * math.sqrt(horizon_hours / _HOURS_PER_YEAR)


def model_move_from_prices(current_price: float, predicted_target_price: float) -> float:
    """Znakowany ułamkowy ruch modelu: `(target - current) / current`.

    Wygodna granica konwersji: wołający podaje `float(current)` i `float(target)`
    (ceny bywają `Decimal` w reszcie systemu). Cena bieżąca 0 → 0.0 (brak punktu
    odniesienia, zamiast dzielenia przez zero).
    """
    if current_price == 0.0:
        return 0.0
    return (predicted_target_price - current_price) / current_price


def evaluate_edge(
    model_move: float,
    iv: float | None,
    horizon_hours: float,
) -> ImpliedEdge:
    """Porównuje znakowany ruch modelu z implikowanym ruchem rynku opcji.

    `model_move` — ułamkowy, znakowany ruch predykcji (np. +0.031 = +3.1%);
    policz go z cen przez `model_move_from_prices`. `iv` — annualizowana IV jako
    ułamek (None = brak danych opcji, np. Finnhub 403). `horizon_hours` — horyzont
    predykcji w godzinach.

    `edge_sigma = |model_move| / implied_move` mierzy WIELKOŚĆ dywergencji (znak
    predykcji jej nie zmienia). Etykieta:
      - `edge_sigma >= 1.5` → MODEL_AHEAD (model widzi więcej, niż rynek wycenił),
      - `edge_sigma <= 0.5` → MARKET_AHEAD (rynek wycenia ruch, model śpi),
      - pomiędzy → ALIGNED.

    Brak IV / `iv <= 0` / `horizon_hours <= 0` → NO_SIGNAL (`edge_sigma` i
    `implied_move` = None, nigdy dzielenia przez zero).
    """
    if iv is None:
        return ImpliedEdge(
            label=EdgeLabel.NO_SIGNAL,
            edge_sigma=None,
            implied_move=None,
            model_move=model_move,
        )

    implied_move = implied_move_12h(iv, horizon_hours)
    if implied_move <= 0.0:
        # iv <= 0 lub horyzont <= 0 — dane bez sensu, brak sygnału.
        return ImpliedEdge(
            label=EdgeLabel.NO_SIGNAL,
            edge_sigma=None,
            implied_move=None,
            model_move=model_move,
        )

    edge_sigma = abs(model_move) / implied_move
    if edge_sigma >= _MODEL_AHEAD_SIGMA:
        label = EdgeLabel.MODEL_AHEAD
    elif edge_sigma <= _MARKET_AHEAD_SIGMA:
        label = EdgeLabel.MARKET_AHEAD
    else:
        label = EdgeLabel.ALIGNED

    return ImpliedEdge(
        label=label,
        edge_sigma=edge_sigma,
        implied_move=implied_move,
        model_move=model_move,
    )
