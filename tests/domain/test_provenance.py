"""Testy domenowych odznak proweniencji (Q6).

`ProvenanceBadge` + `build_provenance_badges` to czysta logika domeny:
zamieniają istniejące sygnały (degraded_reason, wiek fundamentów vs TTL,
data_quality_flags) na etykiety FRESH / STALE / DEGRADED / FLAGGED. Zero I/O,
zero importów zewnętrznych — wejście to proste skalary i listy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.domain.provenance import (
    ProvenanceBadge,
    ProvenanceLevel,
    build_provenance_badges,
)


def _now() -> datetime:
    return datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


class TestProvenanceBadgeValueObject:
    def test_is_frozen(self) -> None:
        badge = ProvenanceBadge(level=ProvenanceLevel.FRESH, label="Świeże", detail=None)
        try:
            badge.label = "inne"  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("ProvenanceBadge powinno być niemutowalne (frozen)")

    def test_carries_level_label_detail(self) -> None:
        badge = ProvenanceBadge(
            level=ProvenanceLevel.STALE,
            label="Dane z cache",
            detail="fundamenty: 30h",
        )
        assert badge.level is ProvenanceLevel.STALE
        assert badge.label == "Dane z cache"
        assert badge.detail == "fundamenty: 30h"


class TestBuildProvenanceBadgesClean:
    def test_clean_inputs_yield_single_fresh_badge(self) -> None:
        # Brak degradacji, fundamenty świeże, brak flag jakości → tylko FRESH.
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=1),
            data_quality_flags=[],
            now=_now(),
        )
        assert len(badges) == 1
        assert badges[0].level is ProvenanceLevel.FRESH

    def test_no_fundamentals_still_fresh_when_nothing_wrong(self) -> None:
        # Brak fundamentów (np. ETF/krypto) nie jest STALE — to po prostu brak
        # tego sygnału. Czyste pozostałe wejścia → FRESH.
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=None,
            data_quality_flags=[],
            now=_now(),
        )
        assert len(badges) == 1
        assert badges[0].level is ProvenanceLevel.FRESH


class TestBuildProvenanceBadgesStale:
    def test_fundamentals_older_than_ttl_is_stale(self) -> None:
        # Wiek fundamentów > TTL (24h) → STALE(cached).
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=30),
            data_quality_flags=[],
            now=_now(),
        )
        levels = [b.level for b in badges]
        assert ProvenanceLevel.STALE in levels
        assert ProvenanceLevel.FRESH not in levels

    def test_fundamentals_within_ttl_not_stale(self) -> None:
        # Dokładnie na granicy TTL (23h < 24h) → wciąż świeże.
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=23),
            data_quality_flags=[],
            now=_now(),
        )
        assert [b.level for b in badges] == [ProvenanceLevel.FRESH]

    def test_custom_ttl_respected(self) -> None:
        # Jawny krótszy TTL przesuwa próg STALE.
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=5),
            data_quality_flags=[],
            now=_now(),
            ttl_hours=4,
        )
        assert any(b.level is ProvenanceLevel.STALE for b in badges)


class TestBuildProvenanceBadgesDegraded:
    def test_degraded_reason_yields_degraded_badge(self) -> None:
        badges = build_provenance_badges(
            degraded_reason="av_keys_exhausted",
            fundamentals_fetched_at=_now() - timedelta(hours=1),
            data_quality_flags=[],
            now=_now(),
        )
        levels = [b.level for b in badges]
        assert ProvenanceLevel.DEGRADED in levels
        # DEGRADED wyklucza FRESH — sygnał jest skażony.
        assert ProvenanceLevel.FRESH not in levels

    def test_degraded_detail_carries_reason(self) -> None:
        badges = build_provenance_badges(
            degraded_reason="av_keys_exhausted",
            fundamentals_fetched_at=None,
            data_quality_flags=[],
            now=_now(),
        )
        degraded = next(b for b in badges if b.level is ProvenanceLevel.DEGRADED)
        assert "av_keys_exhausted" in (degraded.detail or "")

    def test_empty_string_degraded_reason_ignored(self) -> None:
        # Pusty string to nie degradacja — falsy wartość traktujemy jak None.
        badges = build_provenance_badges(
            degraded_reason="",
            fundamentals_fetched_at=None,
            data_quality_flags=[],
            now=_now(),
        )
        assert [b.level for b in badges] == [ProvenanceLevel.FRESH]


class TestBuildProvenanceBadgesFlagged:
    def test_data_quality_flags_yield_flagged_badge(self) -> None:
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=1),
            data_quality_flags=["av_sentiment_score_missing", "news_volume_24h_invalid"],
            now=_now(),
        )
        flagged = [b for b in badges if b.level is ProvenanceLevel.FLAGGED]
        assert len(flagged) == 1
        # Liczba flag widoczna w detalu (audytowalność).
        assert "2" in (flagged[0].detail or "")
        assert ProvenanceLevel.FRESH not in [b.level for b in badges]

    def test_degraded_reason_not_double_counted_as_flag(self) -> None:
        # degraded_reason ląduje też w data_quality_flags (predict_node tak robi).
        # Nie chcemy dwóch odznak o tym samym sensie — DEGRADED ma pierwszeństwo,
        # a FLAGGED liczy tylko POZOSTAŁE flagi.
        badges = build_provenance_badges(
            degraded_reason="av_keys_exhausted",
            fundamentals_fetched_at=None,
            data_quality_flags=["av_keys_exhausted"],
            now=_now(),
        )
        levels = [b.level for b in badges]
        assert ProvenanceLevel.DEGRADED in levels
        assert ProvenanceLevel.FLAGGED not in levels


class TestBuildProvenanceBadgesCombined:
    def test_stale_and_flagged_combine(self) -> None:
        badges = build_provenance_badges(
            degraded_reason=None,
            fundamentals_fetched_at=_now() - timedelta(hours=40),
            data_quality_flags=["trend_direction_invalid"],
            now=_now(),
        )
        levels = {b.level for b in badges}
        assert ProvenanceLevel.STALE in levels
        assert ProvenanceLevel.FLAGGED in levels
        assert ProvenanceLevel.FRESH not in levels
