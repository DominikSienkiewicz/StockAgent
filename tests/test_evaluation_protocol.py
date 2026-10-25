"""WIA-42: maszynowy protokół ewaluacji (`evaluation_protocol.toml`).

Tabela „Pre-registered parameters” w README to deklaracja; hash pliku protokołu
w przepisie i w każdym commitmencie v2 dowodzi, że metryki nie wybrano po
zobaczeniu wyników (HARKing). Dwa źródła prawdy (stałe domeny i plik) — dlatego
te testy pilnują, że plik mówi dokładnie to, co liczy kod.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

import main_agent
from src.application.agent_graph import AgentGraphDeps, _seal_commitment
from src.application.ports import (
    AttestationPublisherPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.application.report_public_scorecard import PROTOCOL_LABELS
from src.application.run_context import RunContext
from src.config import EVALUATION_PROTOCOL_PATH, Settings, load_evaluation_protocol
from src.domain.backtest import LABEL_HORIZON
from src.domain.equity_curve import directional_hit_rate
from src.domain.evaluation_protocol import EvaluationProtocol
from src.domain.prediction import (
    CRYPTO_SIDEWAYS_TOLERANCE,
    PREDICTION_HORIZON_HOURS,
    SIDEWAYS_TOLERANCE,
    Prediction,
    TrendDirection,
)
from src.domain.significance import DEFAULT_ALPHA, DEFAULT_FDR_Q
from src.domain.track_record import NaiveBaselines
from src.domain.value_objects import Threshold
from tests.settings_kwargs import NO_ENV_FILE

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def protocol() -> EvaluationProtocol:
    return load_evaluation_protocol()


def _settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "VOO"],
        ml_model_path="data/models/missing.ubj",
    )


def test_protocol_lives_in_the_repo_root() -> None:
    assert EVALUATION_PROTOCOL_PATH == ROOT / "evaluation_protocol.toml"
    assert EVALUATION_PROTOCOL_PATH.is_file()


def test_protocol_values_match_domain_constants(protocol: EvaluationProtocol) -> None:
    assert protocol.horizon_hours == PREDICTION_HORIZON_HOURS
    assert protocol.horizon_hours == LABEL_HORIZON // timedelta(hours=1)
    # Pasma SIDEWAYS z prediction.py — reguła trafności reflect_node.
    assert protocol.sideways_band == SIDEWAYS_TOLERANCE
    assert protocol.crypto_sideways_band == CRYPTO_SIDEWAYS_TOLERANCE
    assert protocol.alpha == DEFAULT_ALPHA
    assert protocol.fdr_q == DEFAULT_FDR_Q


def test_protocol_window_matches_config_toml(protocol: EvaluationProtocol) -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    assert protocol.window_days == config["track_record_days"]


def test_directional_dead_zone_matches_the_hit_rule(protocol: EvaluationProtocol) -> None:
    # Martwa strefa 0: zmiana dokładnie 0 to pudło i BULLISH, i BEARISH.
    assert protocol.directional_dead_zone == Decimal("0")
    flat = Decimal("100")
    for trend in (TrendDirection.BULLISH, TrendDirection.BEARISH):
        call = Prediction("X", trend, price_at_prediction=flat, predicted_target_price=flat)
        assert call.is_trend_correct(flat) is False
    bullish = Prediction(
        "X", TrendDirection.BULLISH, price_at_prediction=flat, predicted_target_price=flat
    )
    assert bullish.is_trend_correct(flat + Decimal("0.0001")) is True


def test_protocol_dead_zone_is_the_dead_band_of_the_hit_rule(
    protocol: EvaluationProtocol,
) -> None:
    # EXP-06: szerokość martwej strefy żyje w protokole; jego wartość podana jako
    # `dead_band` daje dokładnie regułę, którą liczy dziś reflect (bez pasma).
    flat = Decimal("100")
    for trend in (TrendDirection.BULLISH, TrendDirection.BEARISH):
        call = Prediction("X", trend, price_at_prediction=flat, predicted_target_price=flat)
        for actual in (Decimal("99.99"), flat, Decimal("100.01")):
            assert call.is_trend_correct(
                actual, dead_band=protocol.directional_dead_zone
            ) is call.is_trend_correct(actual)


def test_sideways_stays_out_of_the_directional_hit_rate(protocol: EvaluationProtocol) -> None:
    assert protocol.sideways_in_hit_rate is False
    assert directional_hit_rate([("SIDEWAYS", True)]).directional == 0


def test_naive_baselines_are_the_domain_ones(protocol: EvaluationProtocol) -> None:
    domain = {field.name for field in dataclasses.fields(NaiveBaselines)} - {"n"}
    assert set(protocol.naive_baselines) == domain


def test_every_protocol_name_has_a_public_label(protocol: EvaluationProtocol) -> None:
    names = {
        *protocol.naive_baselines,
        *protocol.benchmarks,
        *protocol.primary_metrics,
        *protocol.secondary_metrics,
    }
    assert names <= PROTOCOL_LABELS.keys()


def test_protocol_hash_is_the_sha256_of_the_file_bytes(protocol: EvaluationProtocol) -> None:
    assert protocol.sha256 == hashlib.sha256(EVALUATION_PROTOCOL_PATH.read_bytes()).hexdigest()


def test_protocol_hash_is_part_of_the_recipe(protocol: EvaluationProtocol) -> None:
    recipe = main_agent.build_prediction_recipe(_settings(), RunContext())

    assert recipe.protocol_sha256 == protocol.sha256
    assert recipe.receipt_fields()["protocol_sha256"] == protocol.sha256


def test_protocol_hash_is_in_commitment_payload(protocol: EvaluationProtocol) -> None:
    recipe = main_agent.build_prediction_recipe(_settings(), RunContext())
    repository = Mock(spec=RepositoryPort)
    repository.get_current_hour_commitment_salt.return_value = None
    deps = AgentGraphDeps(
        market_port=Mock(spec=MarketDataPort),
        sentiment_port=Mock(spec=SentimentPort),
        news_port=Mock(spec=NewsPort),
        repository_port=repository,
        ml_port=Mock(spec=MLPredictionPort),
        llm_port=Mock(spec=LLMPort),
        threshold=Threshold(Decimal("0.02")),
        attestation_publisher=Mock(spec=AttestationPublisherPort),
    )
    record = {
        "symbol": "AAPL",
        "predicted_trend": "BULLISH",
        "predicted_target_price": Decimal("101"),
        "price_at_prediction": Decimal("100"),
        "confidence_score": 0.6,
    }

    _seal_commitment(deps, record, council=None, recipe=recipe)

    payload = json.loads(cast("str", record["commitment_payload"]))
    assert payload["protocol_sha256"] == protocol.sha256
    assert payload["recipe_version"] == recipe.version


def test_another_protocol_is_another_recipe_version() -> None:
    # Zmiana protokołu = nowa wersja przepisu: wyniki liczone starym protokołem
    # zostają w swoim segmencie.
    recipe = main_agent.build_prediction_recipe(_settings(), RunContext())

    changed = dataclasses.replace(recipe, protocol_sha256="0" * 64)

    assert changed.version != recipe.version
