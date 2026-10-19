"""CLI ewaluacja jakości predykcji na podstawie `prediction_logs`.

Mierzy OFFLINE, czy agent faktycznie bije naiwny baseline — zanim zmiana
(próg, model rady, hiperparametry) trafi na produkcję. Read-only, bez kosztów
API: czyta tylko zamknięte (ocenione) predykcje z Supabase.

Metryki:
    - hit_rate     — blended trafność (udział is_trend_correct=True po WSZYSTKICH
                     wierszach; SIDEWAYS liczy się jako trafienie na spokojnych dniach),
    - directional_hit_rate — trafność liczona TYLKO po BULLISH/BEARISH (bez „cichych"
                     trafień SIDEWAYS) + osobne pokrycie SIDEWAYS; definicja z domeny
                     (`directional_hit_rate`), ta sama co w mailu i scorecardzie,
    - evidence     — dowód trafności kierunkowej (WIA-22): naiwne baseline'y na tych
                     samych predykcjach, n_eff po cyklach (dniach UTC) i werdykt testu
                     dwumianowego względem lepszego baseline'u kierunkowego,
    - model_rmse   — RMSE prognozy (predicted_target_price) vs cena rzeczywista,
    - baseline_rmse— RMSE naiwnego "last-value" (cena_w_momencie_predykcji),
    - beats_baseline — czy model_rmse < baseline_rmse (czy w ogóle warto).

Segmentacja (WIA-20): `--by-recipe` dokłada trafność kierunkową per wersja
przepisu predykcji (`prediction_logs.recipe_version`, WYMAGA migracji 030).
`--by-universe` (WIA-48) — per wersja uniwersum symboli z kwitu v2
(`decision_receipts`, bez nowej migracji). Wiersze bez wersji (sprzed WIA-20
albo zapisane bez flagi / bez kwitu v2) to segment `unversioned`.

Uruchom:
    uv run python -m src.tools.evaluate
    uv run python -m src.tools.evaluate --days 60
    uv run python -m src.tools.evaluate --by-recipe
    uv run python -m src.tools.evaluate --by-universe

Exit code: 0 zawsze (narzędzie raportujące, nie bramka CI).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.equity_curve import DirectionalHitRate, directional_hit_rate
from src.domain.recipe import hit_rate_by_version
from src.domain.track_record import HitRateEvidence, ScoredCall, assess_directional_hit_rate

# Prefiks wersji w raporcie CLI — pełny sha256 nic nie mówi człowiekowi.
_VERSION_PREFIX = 12


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
    # WIA-22: przedział na n_eff, baseline'y i werdykt istotności (None bez wierszy).
    evidence: HitRateEvidence | None = None


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _cycle_key(value: Any) -> str:
    """Dzień UTC predykcji jako klucz cyklu; brak/śmieć → wspólny klucz ""."""
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return ""
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return stamp.date().isoformat()


def _scored_call(row: dict[str, Any]) -> ScoredCall:
    price_at = _to_float(row.get("price_at_prediction"))
    actual = _to_float(row.get("actual_price_after_12h"))
    change = None
    if price_at and actual is not None:
        change = (actual - price_at) / price_at
    return ScoredCall(
        cycle=_cycle_key(row.get("timestamp")),
        predicted_trend=str(row.get("predicted_trend")),
        is_correct=bool(row["is_trend_correct"]),
        realized_change=change,
    )


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
    # „dzień bez ruchu", a nie realna trafność kierunku. Jedna definicja z domeny.
    flagged = [r for r in rows if r.get("is_trend_correct") is not None]
    directional = directional_hit_rate(
        (str(r.get("predicted_trend")), bool(r["is_trend_correct"])) for r in flagged
    )
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
        directional_hit_rate=directional.rate,
        directional_count=directional.directional,
        sideways_count=sideways_count,
        evidence=(
            assess_directional_hit_rate([_scored_call(r) for r in flagged]) if flagged else None
        ),
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
        *_evidence_lines(report.evidence),
        f"  RMSE modelu:                    {_num(report.model_rmse)}",
        f"  RMSE baseline (last-value):     {_num(report.baseline_rmse)}",
        f"  Werdykt:                        {verdict}",
    ]
    return "\n".join(lines)


def summarize_by_version(
    rows: list[dict[str, Any]], version_key: str
) -> dict[str, DirectionalHitRate]:
    """Trafność kierunkowa per wartość `version_key` (np. `recipe_version`).

    Bierze tylko wiersze z oceną kierunku; wiersz bez wersji trafia do segmentu
    `UNVERSIONED`. Definicja trafności ta sama co w całym raporcie."""
    return hit_rate_by_version(
        (
            None if r.get(version_key) is None else str(r[version_key]),
            str(r.get("predicted_trend")),
            bool(r["is_trend_correct"]),
        )
        for r in rows
        if r.get("is_trend_correct") is not None
    )


def format_segments(segments: dict[str, DirectionalHitRate], label: str) -> str:
    """Tabela segmentów do CLI: skrócona wersja, n kierunkowych, trafność."""
    lines = [f"  Trafność kierunkowa per {label}:"]
    for version, hit_rate in segments.items():
        rate = "—" if hit_rate.rate is None else f"{hit_rate.rate * 100:.1f}%"
        lines.append(
            f"    {version[:_VERSION_PREFIX]:<{_VERSION_PREFIX}}  {rate:>6}  "
            f"(n={hit_rate.directional}, SIDEWAYS={hit_rate.sideways})"
        )
    return "\n".join(lines)


def _evidence_lines(evidence: HitRateEvidence | None) -> list[str]:
    """Baseline'y i werdykt istotności kierunkowej (puste, gdy brak kierunkowych)."""
    if evidence is None or evidence.hit_rate.rate is None:
        return []
    b = evidence.baselines
    lines = [f"  Efektywna liczność (n_eff):     {evidence.n_eff:.1f} z {evidence.cycles} cykli"]
    if b.always_bullish is not None and b.always_bearish is not None:
        lines.append(
            f"  Naiwne baseline'y:              zawsze-BULLISH: {b.always_bullish * 100:.0f}%"
            f" · zawsze-BEARISH: {b.always_bearish * 100:.0f}%"
        )
    if evidence.verdict is not None:
        lines.append(f"  Istotność (kierunkowa):         {evidence.verdict}")
    return lines


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
    parser.add_argument(
        "--by-recipe",
        action="store_true",
        help="Trafność per wersja przepisu (recipe_version) — wymaga migracji 030.",
    )
    parser.add_argument(
        "--by-universe",
        action="store_true",
        help="Trafność per wersja uniwersum symboli (z kwitu decyzyjnego v2).",
    )
    args = parser.parse_args(argv)

    # Import lokalnie — narzędzie nie powinno wymagać konfiguracji do testów
    # czystej funkcji `summarize_evaluation`.
    from src.config import Settings
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    settings = Settings.from_env()
    repo = SupabaseRepository(url=settings.supabase_url, key=settings.supabase_key)
    # Kolumny segmentów tylko na żądanie: `recipe_version` istnieje dopiero po
    # migracji 030, a domyślny przebieg ma działać na każdej bazie.
    segment_reads = {
        name: True
        for name, wanted in (("with_recipe", args.by_recipe), ("with_universe", args.by_universe))
        if wanted
    }
    rows = repo.get_resolved_predictions_for_eval(days=args.days, **segment_reads)
    report = summarize_evaluation(rows)
    print(format_report(report, args.days))
    if args.by_recipe:
        print(format_segments(summarize_by_version(rows, "recipe_version"), "przepis"))
    if args.by_universe:
        print(format_segments(summarize_by_version(rows, "universe_version"), "uniwersum"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
