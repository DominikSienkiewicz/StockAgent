from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Q3 — Feature attribution (SHAP-lite). Czysta domena: bierze gotowe znakowane
# wkłady per-cecha (policzone przez infrastrukturę z boostera) i porządkuje je
# do prezentacji. Zero importów zewnętrznych — tylko stdlib.

DEFAULT_TOP_N = 5


@dataclass(frozen=True)
class FeatureContribution:
    """Znakowany wkład pojedynczej cechy w predykcję modelu.

    `contribution > 0` pchnęło prognozę w górę, `< 0` w dół. Wartość jest w
    jednostkach targetu modelu (tu: zwrot 12h), zsumowane wkłady + bias dają
    surową predykcję. Czysty value object — render-only w raporcie.
    """

    feature: str
    contribution: float


def top_contributions(
    contribs: Mapping[str, float], top_n: int = DEFAULT_TOP_N
) -> tuple[FeatureContribution, ...]:
    """Zwraca `top_n` cech o NAJWIĘKSZYM module wkładu (najsilniej pchających).

    Ranking po wartości BEZWZGLĘDNEJ malejąco — interesuje nas siła pchnięcia
    (w górę albo w dół), nie sam znak. Sort jest stabilny: przy remisie modułu
    zachowujemy kolejność wstawienia, żeby raport był deterministyczny między
    cyklami. Znak wkładu jest zachowany w zwróconym value object.
    """
    items = [
        FeatureContribution(feature=name, contribution=float(value))
        for name, value in contribs.items()
    ]
    # `sorted` w Pythonie jest stabilny → remisy modułu trzymają insertion order.
    ranked = sorted(items, key=lambda c: abs(c.contribution), reverse=True)
    return tuple(ranked[:top_n])
