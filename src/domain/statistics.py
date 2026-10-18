"""Przedziały ufności dla odsetków trafień (WIA-32) — czysty stdlib.

Goły procent trafności jest nieweryfikowalny: 52% z 10 predykcji i 52% z 1000
to zupełnie inne dowody. Przedział Wilsona (w odróżnieniu od przybliżenia
Walda) nie wychodzi poza [0, 1] i nie zapada się do zerowej szerokości przy
0 albo n trafieniach, więc nadaje się też dla małych prób z raportu.
"""

from __future__ import annotations

import math

# Kwantyl rozkładu normalnego dla dwustronnego przedziału 95%.
Z_95 = 1.959963984540054


def wilson_interval(successes: float, n: float, z: float = Z_95) -> tuple[float, float]:
    """Przedział ufności Wilsona dla odsetka `successes / n`.

    Przyjmuje liczby ułamkowe, bo przy skorelowanych próbach liczy się go na
    efektywnej liczności (`n_eff`, WIA-23): `successes = rate * n_eff`.
    Pusta próba (`n <= 0`) daje maksymalną niepewność `(0.0, 1.0)`.
    """
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, successes / n))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)
