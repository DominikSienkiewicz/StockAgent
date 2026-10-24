"""ARCH-06: składanie kontekstu raportu Fast Loopa z darmowych odczytów.

`ReportAssembler` dokłada do `ReportContext` sekcje liczone z repozytorium:
accuracy, zamknięte predykcje, alerty kwot, historię głosów rady i ranking
person, Track Record (krzywa, kalibracja, lekcje, statystyki, chipy per
symbol), kalibrację rankingu sygnałów, kartę kondycji modelu, zmiany
nastawienia rady oraz dojrzałość cyklu i jego skład sektorowy. Zero płatnych
portów; każdy odczyt best-effort — błąd chowa swoją sekcję, cykl leci dalej.

Zna tylko `RepositoryPort` i `ReportFlags` — application nie importuje
`src.config`, więc `Settings` mapuje na flagi composition root (`main_agent`).
Sekcje z adapterów (portfel właściciela, kursy NBP) i konfigurację prezentacji
(mapa sektorów, strefa czasu) composition root dokłada wcześniej w `context`;
renderem i wysyłką zajmuje się `main_agent.deliver_report` (ARCH-08).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.application.errors import PortError
from src.application.move_interval import interval_coverage
from src.application.ports import RepositoryPort, StoredShockAlert
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext, SymbolResult
from src.application.report_council_history import InvestorHistory
from src.application.report_evidence import (
    PaperCosts,
    accuracy_drop_alert,
    accuracy_drop_evidence,
    hit_rate_evidence,
    portfolio_curve,
)
from src.application.report_lessons import LessonsReport, build_lessons
from src.application.report_model_scorecard import ModelScorecardView, build_scorecard_view
from src.application.report_models import ResolvedPrediction
from src.application.report_signals import parse_resolved_predictions
from src.application.report_symbol_record import SymbolTrackRecordView, symbol_track_record_view
from src.application.report_track_record import TrackRecordStats
from src.application.track_record_window import (
    TrackRecordReads,
    window_benchmark,
    window_coverage,
)
from src.domain.calibration_curve import (
    CalibrationBucket,
    confidence_reorders_ranking,
    discrimination_evidence,
    probability_scores,
    reliability_curve,
)
from src.domain.consensus_shift import ConsensusShift, detect_shift
from src.domain.coverage import CoverageReport
from src.domain.cycle_maturity import CycleMaturity, SkipReason, classify_cycle
from src.domain.equity_curve import EquityCurve
from src.domain.model_scorecard import MODEL_SCORECARD_WINDOW_DAYS
from src.domain.persona_track_record import (
    PersonaTrackRecord,
    assess_against_track_record,
    rank_personas,
    sort_personas,
)
from src.domain.quota import QuotaAlert

ACCURACY_STATS_DAYS = 30
RESOLVED_PREDICTIONS_HOURS = 24
# NIEZ-01: banner alertów kwot sięga tydzień + 3 h wstecz, żeby niedzielny CRITICAL
# ze Slow Loopa (cron raz w tygodniu) był w najbliższym mailu, a nie znikał po 24 h.
QUOTA_ALERT_LOOKBACK_HOURS = 7 * 24 + 3
# ALR-06: okno szoków, gdy nie wiadomo, kiedy wyszedł ostatni mail (doba = cykl
# Fast Loopa w dzień roboczy).
SHOCK_FALLBACK_LOOKBACK_HOURS = 24
# Okno track recordu person (#3) — 90 dni, tak jak domyślne okno RPC z migracji
# 018. Dłuższe niż okno accuracy: hit-rate per persona potrzebuje próbki.
PERSONA_TRACK_RECORD_DAYS = 90

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReportFlags:
    """Flagi i okna sekcji raportu, mapowane z `Settings` w composition root.

    Wartości przechodzą bez konwersji — każda sekcja liczy z nich dokładnie to,
    co liczyła z `Settings`. Flagi sekcji domyślnie wyłączone.
    """

    track_record_days: int
    reflection_min_age_hours: int
    symbols: Sequence[str] = ()
    crypto_symbols: Sequence[str] = ()
    equity_curve_enabled: bool = False
    lessons_enabled: bool = False
    calibration_curve_enabled: bool = False
    calibrated_confidence_enabled: bool = False
    interval_coverage_enabled: bool = False
    model_scorecard_enabled: bool = False
    cycle_diff_enabled: bool = False
    settled_at_windows_enabled: bool = False
    # ALR-06: szoki z watcha (#11) od ostatniego maila — czytane tylko, gdy watch działa.
    shock_alerts_enabled: bool = False


def portfolio_sectors(
    results: list[SymbolResult], sector_map: Mapping[str, str]
) -> dict[str, int]:
    """#4 — skład portfela wg sektorów (sektor → liczba instrumentów).

    Etykiety z mapy sektorów configu (PORT-05); symbole spoza mapy trafiają do
    "Inne", żeby licznik zawsze sumował się do liczby instrumentów.
    """
    counts: dict[str, int] = {}
    for r in results:
        sector = sector_map.get(r.symbol, "Inne")
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def classify_cycle_maturity(results: list[SymbolResult]) -> CycleMaturity:
    """#4 — dojrzałość cyklu z policzonych przyczyn pominięcia.

    Symbole nieobsługiwane cenowo są jawnie POZA mianownikiem (nigdy nie dostaną
    punktu odniesienia), a błędy liczone osobno — cykl zdominowany przez awarie
    źródeł nie może zostać powitaniem "Dzień 1". Reguła siedzi w domenie.
    """
    saved = sum(1 for r in results if r.status == "saved")
    errors = sum(1 for r in results if r.status == "error")
    cold_start = sum(1 for r in results if r.skip_reason is SkipReason.COLD_START)
    below = sum(1 for r in results if r.skip_reason is SkipReason.BELOW_THRESHOLD)
    stale = sum(1 for r in results if r.skip_reason is SkipReason.STALE_REFERENCE)
    # SkipReason.UNSUPPORTED_PRICE celowo nie ma tu swojej liczności — te symbole
    # wypadają z mianownika przez to, że nie trafiają do ŻADNEJ z powyższych sum.
    return classify_cycle(
        saved_count=saved,
        cold_start_count=cold_start,
        below_threshold_count=below,
        error_count=errors,
        stale_reference_count=stale,
    )


def build_consensus_shifts(
    repository: RepositoryPort,
    results: list[SymbolResult],
    *,
    now: datetime,
) -> list[tuple[str, ConsensusShift]]:
    """#8 — wykrywa zmiany nastawienia rady względem poprzedniego cyklu.

    Tłumienie stęchłych flipów: symbol bez ŚWIEŻEGO werdyktu (odcięty bramką
    volatility) jest pomijany — inaczej stary flip pokazywałby się w kółko.
    `gap_hours` liczone z timestampu poprzedniego cyklu; domena degraduje
    porównanie starsze niż 48h do STALE_COMPARISON, zamiast robić dramaturgię.
    Graceful w CALLERZE (styl repo): błąd odczytu → brak sekcji, cykl żyje.
    """
    shifts: list[tuple[str, ConsensusShift]] = []
    for r in results:
        if r.council_verdict is None:
            continue
        try:
            previous = repository.get_previous_verdict(r.symbol)
        except Exception:
            logger.exception("Failed to fetch previous verdict for %s", r.symbol)
            continue
        if previous is None:
            continue
        gap_hours = (now - previous.timestamp).total_seconds() / 3600.0
        shifts.append(
            (
                r.symbol,
                detect_shift(
                    previous.verdict,
                    r.council_verdict,
                    previous_sentiment=previous.sentiment_score,
                    current_sentiment=r.sentiment_score or 0.0,
                    gap_hours=gap_hours,
                ),
            )
        )
    return shifts


def build_model_scorecard(
    flags: ReportFlags, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> ModelScorecardView | None:
    """#12 — widok karty kondycji modelu. Flaga OFF / brak migracji 019 / błąd
    odczytu → None → sekcja sama się chowa, cykl leci dalej. ML-15: okno odczytu
    to `MODEL_SCORECARD_WINDOW_DAYS`, nie `track_record_days` — karta starsza niż
    próg świeżości zostaje widoczna z odznaką STALE."""
    if not flags.model_scorecard_enabled:
        return None
    try:
        rows = repository.get_recent_model_scorecards(MODEL_SCORECARD_WINDOW_DAYS)
    except Exception:
        logger.exception("Failed to fetch model scorecards")
        return None
    view = build_scorecard_view(rows)
    if view is None:
        return None
    # DAT-16: karta mówi też, ile sesji okna ma dane i które symbole nie mają rozliczeń.
    return dataclasses.replace(view, coverage=data_coverage(flags, repository, reads=reads))


def data_coverage(
    flags: ReportFlags, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> CoverageReport | None:
    """DAT-16 — pokrycie danych okna track recordu (`window_coverage`); z `reads`
    (ARCH-16) rozliczenia okna z tego dispatchu zamiast własnego odczytu."""
    return window_coverage(
        repository,
        symbols=flags.symbols,
        crypto_symbols=flags.crypto_symbols,
        days=flags.track_record_days,
        settled_rows=None if reads is None else reads.resolved_rows,
    )


def build_signal_calibration(
    flags: ReportFlags, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> list[CalibrationBucket] | None:
    """#9 — kubełki kalibracji pod RANKING sygnałów (nie pod sekcję Track Record).

    Świadomie osobne od `build_track_record`: `calibration_curve_enabled` rysuje
    sekcję, a `calibrated_confidence_enabled` zmienia ranking. Sprzęgnięcie ich
    znaczyłoby, że włączenie rankingu po cichu dorysowuje sekcję w mailu.
    Best-effort: błąd odczytu → None → surowa pewność, cykl leci dalej.
    """
    if not flags.calibrated_confidence_enabled:
        return None
    window = TrackRecordReads(repository, flags.track_record_days) if reads is None else reads
    try:
        # WIA-39: ranking kalibrujemy krzywą kierunkową — SIDEWAYS trafia za darmo.
        split = window.calibration()
    except Exception:
        logger.exception("Failed to fetch calibration samples for signal ranking")
        return None
    # WIA-38: pewność, która nie odróżnia trafień od pudeł, nie przestawia rankingu.
    discrimination = discrimination_evidence(split.directional_by_cycle, n_boot=0)
    if discrimination is not None and not confidence_reorders_ranking(discrimination):
        logger.warning(
            "Confidence AUC %.2f at n_eff≈%.0f carries no ranking information — "
            "signals ranked on raw confidence",
            discrimination.auc,
            discrimination.n_eff,
        )
        return None
    return reliability_curve(split.directional) if split.directional else None


def build_track_record(
    flags: ReportFlags,
    repository: RepositoryPort,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> tuple[EquityCurve | None, list[CalibrationBucket] | None, LessonsReport | None]:
    """T1/T2/T4 — krzywa kapitału, kalibracja, lessons (render-only, zero
    płatnych). Każda podsekcja flag-gated; wszystkie czytają zamknięte predykcje
    z okna track_record_days. Krzywa kapitału to portfel o równej wadze per
    cykl (WIA-17), nie szeregowe składanie równoległych trade'ów, po kosztach
    obrotu `costs` (WIA-50; None = brutto). `reads` (ARCH-16) — odczyty okna
    z tego dispatchu; None = własne."""
    window = TrackRecordReads(repository, flags.track_record_days) if reads is None else reads
    equity_curve: EquityCurve | None = None
    calibration_buckets: list[CalibrationBucket] | None = None
    lessons: LessonsReport | None = None
    if flags.equity_curve_enabled or flags.lessons_enabled:
        try:
            tr_resolved = window.resolved()
        except Exception:
            logger.exception("Failed to fetch resolved predictions for track record")
            tr_resolved = []
        if flags.equity_curve_enabled:
            equity_curve = portfolio_curve(tr_resolved, costs)
        if flags.lessons_enabled:
            lessons = build_lessons(tr_resolved)
    if flags.calibration_curve_enabled:
        try:
            samples = window.calibration().directional
            calibration_buckets = reliability_curve(samples) if samples else None
        except Exception:
            logger.exception("Failed to compute calibration curve")
    return equity_curve, calibration_buckets, lessons


def build_symbol_track_record(
    flags: ReportFlags, repository: RepositoryPort, *, reads: TrackRecordReads | None = None
) -> SymbolTrackRecordView | None:
    """WIA-35: trafność kierunkowa per symbol do chipu przy predykcji i sygnale.

    Ta sama flaga i to samo okno co krzywa kapitału; darmowy odczyt zamkniętych
    predykcji, zero płatnych wywołań. Błąd odczytu → mail bez chipów."""
    if not flags.equity_curve_enabled:
        return None
    window = TrackRecordReads(repository, flags.track_record_days) if reads is None else reads
    try:
        resolved = window.resolved()
    except Exception:
        logger.exception("Failed to fetch resolved predictions for per-symbol track record")
        return None
    return symbol_track_record_view(resolved, flags.track_record_days)


def build_track_record_stats(
    flags: ReportFlags,
    repository: RepositoryPort,
    costs: PaperCosts | None = None,
    *,
    reads: TrackRecordReads | None = None,
) -> TrackRecordStats | None:
    """Dowód trafności (T1), Brier / BSS (T2) i pokrycie przedziałów 80% (ML-34)
    pod sekcją Track Record.

    Te same flagi co podsekcje, które statystyki uzupełniają; pokrycie za
    `interval_coverage_enabled` (kolumny z migracji 032). Osobna funkcja,
    bo `build_track_record` zwraca stały kontrakt trzech podsekcji. Każdy
    odczyt best-effort: błąd → brak danej statystyki, cykl leci dalej.
    `costs` (WIA-50) to te same założenia, z którymi `build_track_record`
    liczy krzywą — jadą do etykiety „paper-PnL (koszt X bps)” i do porównania
    z koszykiem i IVV (WIA-21), które bez kosztów nie powstaje. `reads`
    (ARCH-16) — odczyty okna z tego dispatchu; None = własne.
    """
    window = TrackRecordReads(repository, flags.track_record_days) if reads is None else reads
    evidence = None
    scores = None
    benchmark = None
    accuracy_drop = None
    if flags.equity_curve_enabled:
        resolved: list[ResolvedPrediction] = []
        try:
            resolved = window.resolved()
            evidence = hit_rate_evidence(resolved)
            # NIEZ-20: data istotnego spadku trafności (Page-Hinkley per cykl).
            accuracy_drop = accuracy_drop_evidence(resolved)
        except Exception:
            logger.exception("Failed to compute hit-rate evidence for track record")
        if resolved and costs is not None:
            benchmark = window_benchmark(
                repository,
                resolved,
                costs,
                window_days=flags.track_record_days,
                settle_after_hours=flags.reflection_min_age_hours,
            )
    sideways_scores = None
    if flags.calibration_curve_enabled:
        try:
            split = window.calibration()
            scores = probability_scores(split.directional)
            # WIA-39: SIDEWAYS osobno — przy małym n wystarczą n i Brier.
            sideways_scores = probability_scores(split.sideways)
        except Exception:
            logger.exception("Failed to compute probability scores for track record")
    coverage = None
    if flags.interval_coverage_enabled:
        try:
            measured = interval_coverage(
                repository.get_settled_moves(flags.track_record_days, with_intervals=True)
            )
            # Bez rozliczonego przedziału nie ma czego mierzyć — podsekcja się chowa.
            coverage = measured if measured.n else None
        except Exception:
            logger.exception("Failed to measure interval coverage for track record")
    paper_costs = costs if flags.equity_curve_enabled else None
    parts = (evidence, scores, coverage, paper_costs, benchmark, accuracy_drop)
    if all(part is None for part in parts):
        return None
    return TrackRecordStats(
        hit_rate=evidence,
        scores=scores,
        coverage=coverage,
        costs=paper_costs,
        benchmark=benchmark,
        sideways_scores=sideways_scores,
        accuracy_drop=accuracy_drop,
    )


def shocks_since_last_mail(
    flags: ReportFlags, repository: RepositoryPort, started_at: datetime
) -> list[StoredShockAlert] | None:
    """ALR-06 — szoki z watcha od startu ostatniego Fast Loopa, którego mail wyszedł.

    Bez takiego przebiegu (pierwszy mail, pauza `cycle_runs`, błąd odczytu) okno to
    ostatnie `SHOCK_FALLBACK_LOOKBACK_HOURS` — wolimy pominąć starszy szok niż
    powtórzyć ten, który był już w mailu. Flaga watcha off → None bez zapytań; błąd
    odczytu alertów → None, sekcja się chowa, cykl leci dalej."""
    if not flags.shock_alerts_enabled:
        return None
    since: datetime | None = None
    try:
        since = repository.get_last_report_at(started_at)
    except PortError:
        logger.warning("Failed to read the last delivered mail — shocks of the last day")
    if since is None:
        since = started_at - timedelta(hours=SHOCK_FALLBACK_LOOKBACK_HOURS)
    try:
        return repository.get_shock_alerts_since(since)
    except PortError:
        logger.warning("Failed to fetch shock alerts — report without them", exc_info=True)
        return None


def persist_quota_alerts(repository: RepositoryPort, alerts: list[QuotaAlert]) -> None:
    """Zapis alertów kwot, każdy osobno — błąd jednego nie gubi pozostałych."""
    for alert in alerts:
        try:
            repository.save_quota_alert(alert)
        except Exception:
            logger.exception("Failed to persist quota alert from %s", alert.source)


def fetch_report_context(
    repository: RepositoryPort,
    quota_monitor: QuotaMonitor,
    all_symbols: list[str],
    *,
    resolved_since: datetime | None = None,
) -> tuple[
    dict[str, Any] | None,
    list[ResolvedPrediction] | None,
    list[QuotaAlert],
    list[InvestorHistory],
    list[PersonaTrackRecord],
]:
    """Best-effort dociągnięcie danych do raportu: accuracy, zamknięte predykcje,
    alerty kwot (persystencja bieżących + odczyt z ostatniego tygodnia), historia głosów rady
    i ranking wiarygodności person.
    Każdy odczyt osobno owinięty — błąd jednego nie blokuje pozostałych.

    `resolved_since` (WIA-19) — zamknięte predykcje to te ROZLICZONE od tej
    chwili (start cyklu), a nie utworzone w ostatnich 24 h. None = okno 24 h.

    MAIL-25: nieudany odczyt zamkniętych predykcji zwraca None, nie pustą listę —
    pusta lista znaczy w mailu „brak rozliczeń”, czego po awarii nie wiemy."""
    accuracy_stats: dict[str, Any] | None = None
    resolved: list[ResolvedPrediction] | None = None
    try:
        accuracy_stats = repository.get_accuracy_stats(ACCURACY_STATS_DAYS)
    except Exception:
        logger.exception("Failed to fetch accuracy stats — report without them")
    try:
        resolved_rows = (
            repository.get_recently_resolved_predictions(RESOLVED_PREDICTIONS_HOURS)
            if resolved_since is None
            else repository.get_recently_resolved_predictions(
                RESOLVED_PREDICTIONS_HOURS, since=resolved_since
            )
        )
        resolved = parse_resolved_predictions(resolved_rows)
    except Exception:
        logger.exception("Failed to fetch resolved predictions")
        resolved = None

    # Persystencja alertów z BIEŻĄCEGO cyklu (przed odczytem history, żeby też
    # tu trafiły) + odczyt z okna bannera.
    persist_quota_alerts(repository, quota_monitor.alerts)
    try:
        recent_alerts = repository.get_recent_quota_alerts(hours=QUOTA_ALERT_LOOKBACK_HOURS)
    except Exception:
        logger.exception("Failed to fetch recent quota alerts — using cycle-only")
        recent_alerts = list(quota_monitor.alerts)

    # U5: historia głosów rady per inwestor (panel drill-down w raporcie).
    council_history: list[InvestorHistory] = []
    try:
        council_history = repository.get_council_vote_history(
            all_symbols, window_days=30
        )
    except Exception:
        logger.exception("Failed to fetch council vote history")

    # #3: leaderboard person. Surowe (hit_rate, votes) z RPC, ranking i próg
    # min_votes liczy domena. Brak migracji 018 / błąd RPC → pusty ranking
    # → sekcja sama się chowa, cykl leci dalej. WIA-30: po dolnej granicy Wilsona.
    persona_track_record: list[PersonaTrackRecord] = []
    try:
        persona_track_record = rank_personas(
            repository.get_persona_track_record(PERSONA_TRACK_RECORD_DAYS), rank_by="wilson"
        )
    except Exception:
        logger.exception("Failed to fetch persona track record")
    return (
        accuracy_stats,
        resolved,
        recent_alerts,
        council_history,
        persona_track_record,
    )


def assess_personas(
    records: list[PersonaTrackRecord], stats: TrackRecordStats | None
) -> list[PersonaTrackRecord]:
    """WIA-22: werdykt istotności person (BH-FDR) na parametrach z okna scorecardu,
    bez migracji; WIA-30: ranking po dolnej granicy Wilsona na n_eff — tym samym
    przedziale, który pokazuje wiersz. Bez dowodu trafności okna
    (`equity_curve_enabled` off, brak danych) ranking zostaje bez werdyktu."""
    evidence = None if stats is None else stats.hit_rate
    if not records or evidence is None:
        return records
    return sort_personas(assess_against_track_record(records, evidence), "wilson")


@dataclass(frozen=True)
class AssembledReport:
    """Kontekst raportu cyklu gotowy do renderu i wysyłki oraz odczyty okna
    track recordu i koszty paper-PnL, które po wysyłce bierze publiczny
    scorecard (ARCH-16, WIA-50)."""

    context: ReportContext
    reads: TrackRecordReads
    costs: PaperCosts


class ReportAssembler:
    """ARCH-06: składa `ReportContext` cyklu z darmowych odczytów repozytorium.

    `costs` (WIA-50) to założenia kosztowe paper-PnL — te same dla krzywej,
    jej etykiety i porównania z benchmarkami.
    """

    def __init__(self, repository: RepositoryPort, flags: ReportFlags, costs: PaperCosts) -> None:
        self._repository = repository
        self._flags = flags
        self._costs = costs

    def assemble(
        self,
        *,
        results: list[SymbolResult],
        started_at: datetime,
        quota_monitor: QuotaMonitor,
        all_symbols: list[str],
        context: ReportContext,
    ) -> AssembledReport:
        """Dokłada do `context` sekcje z repozytorium.

        `context` przynosi sekcje policzone przed wysyłką (ryzyko makro, radar,
        alfa, koszt cyklu), portfel właściciela i konfigurację prezentacji;
        skład sektorowy liczy z jego `sector_map`. Alerty kwot bieżącego cyklu
        zapisuje `fetch_report_context`; wyjątek spoza sekcji leci do wołającego.
        """
        flags = self._flags
        repository = self._repository
        # WIA-19: za flagą sekcja zamkniętych predykcji = rozliczone w TYM cyklu
        # (`settled_at` ≥ start cyklu, migracja 029). Trigger stempluje zegarem
        # bazy, a start liczy zegar runnera: oba chodzą po NTP, a od startu do
        # pierwszego rozliczenia mijają sekundy, więc rozjazd zegarów nie gubi wierszy.
        resolved_since = started_at if flags.settled_at_windows_enabled else None
        (
            accuracy_stats,
            resolved,
            recent_alerts,
            council_history,
            persona_track_record,
        ) = fetch_report_context(
            repository, quota_monitor, all_symbols, resolved_since=resolved_since
        )

        # T1/T2/T4: Track Record (render-only, zero płatnych). Każda podsekcja
        # flag-gated; wszystkie czytają zamknięte predykcje z okna track_record_days.
        # ARCH-16: rozliczenia i próbki kalibracji okna — jeden odczyt na dispatch.
        reads = TrackRecordReads(repository, flags.track_record_days)
        equity_curve, calibration_buckets, lessons = build_track_record(
            flags, repository, self._costs, reads=reads
        )
        track_record_stats = build_track_record_stats(
            flags, repository, self._costs, reads=reads
        )
        persona_track_record = assess_personas(persona_track_record, track_record_stats)
        # NIEZ-20: alarm detektora trafności do bannera tego maila (bez zapisu —
        # liczony na nowo w każdym cyklu, więc nie dubluje wierszy `quota_alerts`).
        if track_record_stats is not None and track_record_stats.accuracy_drop is not None:
            recent_alerts = [
                *recent_alerts,
                accuracy_drop_alert(track_record_stats.accuracy_drop, started_at),
            ]
        # #9: kubełki pod ranking sygnałów (osobna flaga, ten sam odczyt okna).
        signal_calibration_buckets = build_signal_calibration(flags, repository, reads=reads)
        # #4: dojrzałość cyklu + skład sektorowy pod powitanie "Dzień 1".
        maturity = classify_cycle_maturity(results)
        sectors = portfolio_sectors(results, context.sector_map or {})
        # #12: karta kondycji modelu (flaga + migracja 019).
        model_scorecard = build_model_scorecard(flags, repository, reads=reads)
        # #8: zmiany nastawienia rady vs poprzedni cykl (flaga, bez migracji).
        consensus_shifts = (
            build_consensus_shifts(repository, results, now=started_at)
            if flags.cycle_diff_enabled
            else []
        )
        report_context = dataclasses.replace(
            context,
            accuracy_stats=accuracy_stats,
            resolved_predictions=resolved,
            resolved_this_cycle=resolved_since is not None,
            resolved_unavailable=resolved is None,
            council_history=council_history,
            persona_track_record=persona_track_record,
            quota_alerts=recent_alerts,
            equity_curve=equity_curve,
            calibration_buckets=calibration_buckets,
            lessons=lessons,
            track_record_stats=track_record_stats,
            symbol_track_record=build_symbol_track_record(flags, repository, reads=reads),
            signal_calibration_buckets=signal_calibration_buckets,
            cycle_maturity=maturity,
            portfolio_sectors=sectors,
            model_scorecard=model_scorecard,
            consensus_shifts=consensus_shifts,
            shock_alerts=shocks_since_last_mail(flags, repository, started_at),
        )
        return AssembledReport(context=report_context, reads=reads, costs=self._costs)
