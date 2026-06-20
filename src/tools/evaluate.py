"""CLI ewaluacja jakości predykcji na podstawie `prediction_logs`.

Mierzy OFFLINE, czy agent faktycznie bije naiwny baseline — zanim zmiana
(próg, model rady, hiperparametry) trafi na produkcję. Read-only, bez kosztów
API: czyta tylko zamknięte (ocenione) predykcje z Supabase.

Metryki:
    - hit_rate     — blended trafność (udział is_trend_correct=True po WSZYSTKICH
                     wierszach; SIDEWAYS liczy się jako trafienie na spokojnych dniach),
    - directional_hit_rate — trafność liczona TYLKO po BULLISH/BEARISH (bez „cichych"
                     trafień SIDEWAYS) + osobne pokrycie SIDEWAYS,
    - model_rmse   — RMSE prognozy (predicted_target_price) vs cena rzeczywista,
    - baseline_rmse— RMSE naiwnego "last-value" (cena_w_momencie_predykcji),
    - beats_baseline — czy model_rmse < baseline_rmse (czy w ogóle warto).

Uruchom:
    uv run python -m src.tools.evaluate
    uv run python -m src.tools.evaluate --days 60

Exit code: 0 zawsze (narzędzie raportujące, nie bramka CI).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalReport:
    sample_count: int
    hit_rate: float | None        # blended trafność (BULLISH/BEARISH/SIDEWAYS)
    model_rmse: float | None      # RMSE prognozy vs rzeczywistość
    baseline_rmse: float | None   # RMSE naiwnego "brak zmiany"
    beats_baseline: bool          # czy model bije baseline
    # Finding #24: blended hit_rate zalicza SIDEWAYS jako trafienie zawsze, gdy
    # ruch < ±0.5% — banking trafień na spokojnych dniach myli „trafiłem kierunek"
    # z „nic się nie ruszyło". Rozdzielamy directional-only od pokrycia SIDEWAYS.
    directional_hit_rate: float | None  # trafność tylko BULLISH/BEARISH
    directional_count: int              # liczba kierunkowych z flagą
    sideways_count: int                 # liczba SIDEWAYS z flagą


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _rmse(squared_errors: list[float]) -> float | None:
    if not squared_errors:
        return None
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def summarize_evaluation(rows: list[dict[str, Any]]) -> EvalReport:
    """Liczy metryki ewaluacji z zamkniętych predykcji (czysta funkcja).

    `rows` to rekordy `prediction_logs` z kluczami: `predicted_trend`,
    `is_trend_correct`, `price_at_prediction`, `predicted_target_price`,
    `actual_price_after_12h`. Wiersze bez kompletu cen są pomijane przy RMSE,
    ale wciąż liczą się do hit-rate (jeśli mają `is_trend_correct`).
    """
    flags = [
        bool(r["is_trend_correct"])
        for r in rows
        if r.get("is_trend_correct") is not None
    ]
    hit_rate = (sum(flags) / len(flags)) if flags else None

    # Directional-only hit-rate: tylko predykcje BULLISH/BEARISH. SIDEWAYS jest
    # raportowane jako osobne pokrycie (count), bo trafienie SIDEWAYS to często
    # „dzień bez ruchu", a nie realna trafność kierunku.
    directional_flags = [
        bool(r["is_trend_correct"])
        for r in rows
        if r.get("is_trend_correct") is not None
        and str(r.get("predicted_trend")).upper() in ("BULLISH", "BEARISH")
    ]
    directional_hit_rate = (
        (sum(directional_flags) / len(directional_flags))
        if directional_flags
        else None
    )
    directional_count = len(directional_flags)
    sideways_count = sum(
        1
        for r in rows
        if r.get("is_trend_correct") is not None
        and str(r.get("predicted_trend")).upper() == "SIDEWAYS"
    )

    model_errors: list[float] = []
    baseline_errors: list[float] = []
    for r in rows:
        actual = _to_float(r.get("actual_price_after_12h"))
        if actual is None:
            continue
        predicted = _to_float(r.get("predicted_target_price"))
        price_at = _to_float(r.get("price_at_prediction"))
        if predicted is not None:
            model_errors.append((predicted - actual) ** 2)
        if price_at is not None:
            baseline_errors.append((price_at - actual) ** 2)

    model_rmse = _rmse(model_errors)
    baseline_rmse = _rmse(baseline_errors)
    beats_baseline = (
        model_rmse is not None
        and baseline_rmse is not None
        and model_rmse < baseline_rmse
    )
    return EvalReport(
        sample_count=len(rows),
        hit_rate=hit_rate,
        model_rmse=model_rmse,
        baseline_rmse=baseline_rmse,
        beats_baseline=beats_baseline,
        directional_hit_rate=directional_hit_rate,
        directional_count=directional_count,
        sideways_count=sideways_count,
    )


def format_report(report: EvalReport, days: int) -> str:
    """Renderuje EvalReport do czytelnego tekstu (CLI)."""

    def _pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    def _num(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "—"

    verdict = "✅ bije baseline" if report.beats_baseline else "⚠️ NIE bije baseline"
    directional = (
        f"{_pct(report.directional_hit_rate)} "
        f"(n={report.directional_count}, SIDEWAYS={report.sideways_count})"
    )
    lines = [
        f"StockAgent — ewaluacja predykcji (ostatnie {days} dni)",
        "=" * 56,
        f"  Próbek (zamkniętych predykcji): {report.sample_count}",
        f"  Hit-rate (blended):             {_pct(report.hit_rate)}",
        f"  Hit-rate (kierunkowy):          {directional}",
        f"  RMSE modelu:                    {_num(report.model_rmse)}",
        f"  RMSE baseline (last-value):     {_num(report.baseline_rmse)}",
        f"  Werdykt:                        {verdict}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate StockAgent prediction quality vs a naive baseline.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Okno (dni) zamkniętych predykcji do ewaluacji (default: 30).",
    )
    args = parser.parse_args(argv)

    # Import lokalnie — narzędzie nie powinno wymagać konfiguracji do testów
    # czystej funkcji `summarize_evaluation`.
    from src.config import Settings
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    settings = Settings.from_env()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    rows = repo.get_resolved_predictions_for_eval(days=args.days)
    report = summarize_evaluation(rows)
    print(format_report(report, args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
