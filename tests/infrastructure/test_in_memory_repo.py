"""DX-04: `InMemoryRepository` — `RepositoryPort` na słownikach dla trybu offline.

Fake ma semantykę Supabase tam, gdzie graf i raport na niej stoją: upsert
predykcji i snapshotu ceny po (symbol, godzina UTC), kolejka nierozliczonych
od najstarszej z progiem wieku, rozliczenie tylko raz, okna odczytów liczone
od wstrzykniętego zegara i te same filtry trafności (atrapy LLM poza próbą).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from src.application.ports import RepositoryPort
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import Money
from src.infrastructure.adapters.in_memory_repo import InMemoryRepository

START = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)


class _Clock:
    """Zegar testu, który przesuwamy ręcznie."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, hours: float) -> None:
        self.now += timedelta(hours=hours)


def _record(symbol: str = "NVDA", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "symbol": symbol,
        "price_at_prediction": Decimal("100.00"),
        "predicted_target_price": Decimal("103.00"),
        "predicted_trend": "BULLISH",
        "confidence_score": 0.7,
        "reasoning_text": "Popyt na akceleratory.",
        "news_summary": "Nagłówek",
        "council_verdict": None,
        "data_quality_flags": [],
    }
    record.update(overrides)
    return record


def _repo(clock: Callable[[], datetime]) -> InMemoryRepository:
    return InMemoryRepository(clock=clock)


def test_implements_every_abstract_method_of_the_port() -> None:
    repo = _repo(_Clock())

    assert isinstance(repo, RepositoryPort)
    assert not InMemoryRepository.__abstractmethods__


def test_save_prediction_upserts_per_symbol_and_hour() -> None:
    clock = _Clock()
    repo = _repo(clock)

    first = repo.save_prediction(_record(predicted_trend="BEARISH"))
    clock.advance(0.25)
    same_hour = repo.save_prediction(_record(predicted_trend="BULLISH"))
    clock.advance(1)
    next_hour = repo.save_prediction(_record())

    assert same_hour == first
    assert next_hour != first
    unverified = repo.get_unverified_prediction("NVDA")
    assert unverified is not None
    assert unverified.id == first
    assert unverified.predicted_trend == "BULLISH"
    # Upsert nie przesuwa chwili zapisu pierwszego wiersza (kolumna z DEFAULT NOW()).
    assert unverified.created_at == START


def test_price_snapshot_upserts_per_symbol_and_hour() -> None:
    clock = _Clock()
    repo = _repo(clock)

    repo.save_price_snapshot("NVDA", Money(Decimal("100")))
    clock.advance(0.25)
    repo.save_price_snapshot("NVDA", Money(Decimal("101")))
    clock.advance(12)
    repo.save_price_snapshot("NVDA", Money(Decimal("104")))

    history = repo.get_price_history("NVDA", days=2)
    assert [price.amount for _, price in history] == [Decimal("101"), Decimal("104")]
    assert repo.get_last_price("NVDA") == Money(Decimal("104"))
    assert repo.get_last_price_snapshot("NVDA") == (
        Money(Decimal("104")),
        START + timedelta(hours=12.25),
    )
    assert repo.get_last_price("AAPL") is None


def test_unverified_queue_is_oldest_first_and_respects_minimum_age() -> None:
    clock = _Clock()
    repo = _repo(clock)
    oldest = repo.save_prediction(_record())
    clock.advance(12)
    repo.save_prediction(_record(price_at_prediction=Decimal("103")))

    assert repo.get_unverified_prediction("NVDA", min_age_hours=13) is None
    found = repo.get_unverified_prediction("NVDA", min_age_hours=12)
    assert found is not None
    assert found.id == oldest
    assert found.price_at_prediction == Decimal("100.00")


def test_settlement_closes_a_prediction_once_and_feeds_the_report_reads() -> None:
    clock = _Clock()
    repo = _repo(clock)
    prediction_id = repo.save_prediction(_record())
    clock.advance(12)

    repo.update_prediction_accuracy(
        prediction_id, Decimal("103.00"), 1.0, True, "Trafiona predykcja."
    )
    repo.update_prediction_accuracy(
        prediction_id, Decimal("90.00"), 0.1, False, "Druga ocena nie wchodzi."
    )

    assert repo.get_unverified_prediction("NVDA") is None
    resolved = repo.get_recently_resolved_predictions(24)
    assert [(row["symbol"], row["is_trend_correct"]) for row in resolved] == [("NVDA", True)]
    assert Decimal(str(resolved[0]["actual_price_after_12h"])) == Decimal("103.00")
    settled_now = repo.get_recently_resolved_predictions(24, since=clock())
    assert len(settled_now) == 1
    stats = repo.get_accuracy_stats(30)
    assert (stats["sample_count"], stats["correct_count"], stats["mean_accuracy"]) == (1, 1, 1.0)


def test_llm_fallback_rows_stay_out_of_the_hit_rate() -> None:
    clock = _Clock()
    repo = _repo(clock)
    dummy = repo.save_prediction(_record(data_quality_flags=["llm_analysis_failed"]))
    clock.advance(12)
    repo.update_prediction_accuracy(dummy, Decimal("103"), 1.0, True, "ok")

    stats = repo.get_accuracy_stats(30)

    assert stats["sample_count"] == 0
    assert stats["mean_accuracy"] is None
    assert repo.get_resolved_predictions(30) == []


def test_last_prediction_price_skips_the_hour_the_current_write_overwrites() -> None:
    clock = _Clock()
    repo = _repo(clock)
    repo.save_prediction(_record(price_at_prediction=Decimal("100")))
    clock.advance(12)
    repo.save_prediction(_record(price_at_prediction=Decimal("103")))

    assert repo.get_last_prediction_price("NVDA") == Money(Decimal("103"))
    assert repo.get_last_prediction_price("NVDA", before=clock()) == Money(Decimal("100"))


def test_expired_predictions_leave_the_queue_and_are_counted() -> None:
    clock = _Clock()
    repo = _repo(clock)
    repo.save_prediction(_record())
    clock.advance(100)

    assert repo.expire_unverified_predictions("NVDA", older_than_hours=96) == 1
    assert repo.expire_unverified_predictions("NVDA", older_than_hours=96) == 0
    assert repo.get_unverified_prediction("NVDA") is None
    assert repo.count_expired_unverified(30) == 1


def test_quota_alerts_come_back_newest_first_within_the_window() -> None:
    clock = _Clock()
    repo = _repo(clock)
    old = QuotaAlert("AV", QuotaSeverity.WARNING, "stary", "nic", START - timedelta(hours=30))
    new = QuotaAlert("LLM", QuotaSeverity.CRITICAL, "nowy", "doładuj", START - timedelta(hours=1))
    repo.save_quota_alert(old)
    repo.save_quota_alert(new)

    assert repo.get_recent_quota_alerts(hours=24) == [new]
    assert repo.get_recent_quota_alerts(hours=48) == [new, old]


def _opinion(name: str, recommendation: str) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence=0.6,
        reasoning=f"{name}: uzasadnienie",
        key_factors=("wycena",),
    )


def test_council_votes_round_trip_and_group_into_history() -> None:
    clock = _Clock()
    repo = _repo(clock)
    prediction_id = repo.save_prediction(_record())
    repo.save_council_votes(
        prediction_id, "NVDA", [_opinion("Warren Buffett", "HOLD"), _opinion("Cathie Wood", "BUY")]
    )

    votes = repo.get_council_votes_for_prediction(prediction_id)
    history = repo.get_council_vote_history(["NVDA"], window_days=30)

    assert [(vote.investor_name, vote.recommendation) for vote in votes] == [
        ("Warren Buffett", "HOLD"),
        ("Cathie Wood", "BUY"),
    ]
    assert {(h.investor_name, h.recommendations) for h in history} == {
        ("Warren Buffett", ("HOLD",)),
        ("Cathie Wood", ("BUY",)),
    }


def test_previous_verdict_reads_the_latest_saved_council_verdict() -> None:
    clock = _Clock()
    repo = _repo(clock)
    verdict = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.8,
        summary="Rada za kupnem.",
        dissenting_views=("Warren Buffett: drogo",),
        investor_opinions=(_opinion("Cathie Wood", "BUY"),),
    )
    repo.save_prediction(_record(council_verdict=dataclasses.asdict(verdict), sentiment_score=0.25))

    previous = repo.get_previous_verdict("NVDA")

    assert previous is not None
    assert previous.verdict == verdict
    assert previous.sentiment_score == pytest.approx(0.25)
    assert previous.timestamp == START


def test_llm_cost_since_sums_priced_calls_saved_in_the_window() -> None:
    """FIN-25: okno limitu liczy wiersze ledgera po chwili zapisu (jak
    `llm_calls.created_at`); wiersz bez wyceny nie wchodzi do sumy."""
    from src.domain.finops import LLMCallEntry, LLMUsage

    def entry(cost: float | None) -> LLMCallEntry:
        return LLMCallEntry(
            run_id="local",
            node="predict",
            symbol="NVDA",
            usage=LLMUsage(provider="anthropic", model="m", input_tokens=1, output_tokens=1),
            cost_usd=cost,
            price_as_of=None,
        )

    clock = _Clock()
    repo = _repo(clock)
    repo.save_llm_calls([entry(0.5)])
    clock.advance(hours=2)
    repo.save_llm_calls([entry(0.25), entry(None)])

    assert repo.get_llm_cost_since(START + timedelta(hours=1)) == pytest.approx(0.25)
    assert repo.get_llm_cost_since(START) == pytest.approx(0.75)


def test_variant_trend_row_stays_out_of_the_ml_leg_read() -> None:
    # EXP-26: wiersz wariantu LLM dzieli tabelę z nogą ML, ale odczyt ML-33 go pomija.
    from src.domain.prediction import TrendDirection
    from src.domain.promotion import ShadowForecast
    from src.domain.shadow_prompt import ShadowTrendForecast

    clock = _Clock()
    repo = _repo(clock)
    prediction_id = repo.save_prediction(_record())
    repo.save_shadow_predictions(
        prediction_id, "NVDA", [ShadowForecast("champion", "a" * 64, predicted_return=0.01)]
    )
    repo.save_shadow_trend_prediction(
        prediction_id,
        "NVDA",
        ShadowTrendForecast("llm_variant", "e" * 64, TrendDirection.BEARISH, 0.7),
    )
    clock.advance(12)
    repo.update_prediction_accuracy(prediction_id, Decimal("103.00"), 1.0, True, "Trafiona.")

    settled = repo.get_settled_shadow_predictions(START - timedelta(hours=1))

    assert [row.forecast.model_name for row in settled] == ["champion"]


def test_judged_diagnoses_leave_the_sample_and_a_rejection_hides_the_lesson() -> None:
    # EXP-10: ocena wypisuje diagnozę z próbki, odrzucenie zostawia tekst w
    # `rejected_diagnosis`, a w `correction_insights` placeholder poza lessons i RAG.
    from src.domain.prediction import DIAGNOSIS_REJECTED_INSIGHT

    clock = _Clock()
    repo = _repo(clock)
    kept = repo.save_prediction(_record("NVDA"))
    rejected = repo.save_prediction(_record("AMD"))
    clock.advance(12)
    repo.update_prediction_accuracy(kept, Decimal("90"), 0.1, False, "Zignorowałem Fed.")
    repo.update_prediction_accuracy(rejected, Decimal("90"), 0.1, False, "Rynek był zmienny.")

    sample = repo.get_diagnoses_to_judge(7)
    repo.save_diagnosis_judgement(kept, score=3)
    repo.save_diagnosis_judgement(rejected, score=1, rejected_insight="Inny tekst.")
    repo.save_diagnosis_judgement(rejected, score=1, rejected_insight="Rynek był zmienny.")
    repo.save_diagnosis_judgement("missing", score=0)

    assert sorted(row.insight for row in sample) == ["Rynek był zmienny.", "Zignorowałem Fed."]
    assert repo.get_diagnoses_to_judge(7) == []
    rows = {row["symbol"]: row for row in repo.get_recently_resolved_predictions(24)}
    assert rows["AMD"]["correction_insights"] == DIAGNOSIS_REJECTED_INSIGHT
    assert rows["AMD"]["rejected_diagnosis"] == "Rynek był zmienny."
    assert rows["NVDA"]["correction_insights"] == "Zignorowałem Fed."


def test_shock_alert_levels_follow_the_ladder_rungs() -> None:
    # ALR-16: drugi szczebel dnia mieści się obok pierwszego, ten sam szczebel odbija
    # się jak UNIQUE(symbol, alert_date, level) z migracji 081.
    from datetime import date

    from src.domain.shock import SentShockLevel, ShockDirection

    repo = _repo(_Clock())
    day = date(2026, 10, 24)
    repo.save_shock_alert("NVDA", day, -0.045, "DOWN")
    repo.save_shock_alert("NVDA", day, -0.095, "DOWN", level=2)

    assert repo.get_sent_shock_alerts(day) == {("NVDA", day)}
    assert repo.get_shock_alert_levels(day) == {
        "NVDA": (
            SentShockLevel(level=1, direction=ShockDirection.DOWN),
            SentShockLevel(level=2, direction=ShockDirection.DOWN),
        )
    }
    with pytest.raises(ValueError, match="already exists"):
        repo.save_shock_alert("NVDA", day, -0.12, "DOWN", level=2)


def test_revision_votes_sit_next_to_the_first_round_and_reads_stay_on_round_one() -> None:
    # ML-40: klucz jak UNIQUE(prediction_id, investor_name, round) z migracji 082;
    # odczyty (dissent-replay, historia, redundancja) biorą niezależną rundę 1.
    repo = _repo(_Clock())
    prediction_id = repo.save_prediction(_record())
    repo.save_council_votes(prediction_id, "NVDA", [_opinion("Cathie Wood", "SELL")])
    repo.save_council_votes(prediction_id, "NVDA", [_opinion("Cathie Wood", "BUY")], debate_round=2)

    votes = repo.get_council_votes_for_prediction(prediction_id)
    history = repo.get_council_vote_history(["NVDA"], window_days=30)

    assert [(vote.investor_name, vote.recommendation) for vote in votes] == [
        ("Cathie Wood", "SELL")
    ]
    assert [h.recommendations for h in history] == [("SELL",)]
    assert [s.recommendation for s in repo.get_council_vote_samples()] == ["SELL"]
