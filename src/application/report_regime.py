"""ALR-05 — reżim rynku i mnożnik progu volatility w nagłówku maila.

`regime_aware_enabled` zmienia bramkę każdego cyklu: w RISK_OFF próg rośnie, więc
czytelnik widzi mniej predykcji. Do ALR-05 reżim trafiał wyłącznie do logów, a mail
nie mówił, dlaczego predykcji jest mniej. Render-only, zero I/O.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from decimal import Decimal

from src.domain.number_format import format_number_pl, format_pct_pl
from src.domain.regime import MarketRegime


@dataclass(frozen=True)
class RegimeBanner:
    """Reżim efektywny cyklu (po histerezie), mnożnik progu (≥ 1.0), próg akcji
    przed mnożnikiem (ułamek, `volatility_threshold`) i powód klasyfikacji."""

    regime: MarketRegime
    multiplier: float
    base_threshold: Decimal
    reason: str


def render_regime_line(banner: RegimeBanner) -> str:
    """„Reżim: RISK_OFF · próg ×1,5 (2,0% → 3,0%) · powód: 2 głosy ELEVATED”.

    Bez zacieśnienia (×1,0) jeden próg zamiast strzałki. Liczby po polsku (MAIL-58)."""
    multiplier = Decimal(str(banner.multiplier))
    base = banner.base_threshold * 100
    thresholds = format_pct_pl(base, signed=False)
    if multiplier != 1:
        thresholds += f" → {format_pct_pl(base * multiplier, signed=False)}"
    return (
        f"Reżim: {banner.regime.value} · "
        f"próg ×{format_number_pl(multiplier, signed=False, decimals=1)} ({thresholds}) · "
        f"powód: {banner.reason}"
    )


def render_regime_html(banner: RegimeBanner | None) -> str:
    """Linia reżimu pod czasem startu w nagłówku HTML; "" bez reżimu."""
    if banner is None:
        return ""
    return f"<br/>{html.escape(render_regime_line(banner))}"


def render_regime_text(banner: RegimeBanner | None) -> str:
    """Linia reżimu w nagłówku wersji tekstowej; "" bez reżimu."""
    return render_regime_line(banner) if banner is not None else ""
