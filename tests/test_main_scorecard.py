"""WIA-52: tani entry-point publicznego scorecardu (`main_scorecard.py`) i jego
workflow `scorecard_pages.yml`.

Scorecard to jedyny artefakt, który buduje zaufanie obcych, a Fast Loop stoi
(decyzja 1). Entry-point renderuje stronę z danych, które już są w bazie, i składa
wyłącznie repozytorium oraz publisher scorecardu — żadnego portu LLM, Alpha
Vantage ani embeddingów. Kontrakt deployu Pages (katalog scorecardu, osobny job
deploy) pilnuje `tests/test_workflow_env_wiring.py`.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

import main_scorecard
from src.application.ports import RepositoryPort, WebPublisherPort
from src.config import Settings
from tests.settings_kwargs import NO_ENV_FILE

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "scorecard_pages.yml"
ENTRYPOINT = "main_scorecard.py"

# Fabryki płatnych portów z DI Containera Fast Loopa — scorecard nie woła żadnej.
PAID_BUILDERS = (
    "build_llm_adapter",
    "build_tool_use_adapter",
    "build_council_adapter",
    "build_council_llm_adapter",
    "build_embedding_adapter",
    "build_use_case",
    "AlphaVantageClient",
)
PAID_SECRET_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FINNHUB_API_KEY")
FREE_SECRET_KEYS = {"SUPABASE_URL", "SUPABASE_KEY"}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["NVDA", "AMD"],
        crypto_symbols=[],
        volatility_threshold=Decimal("0.02"),
        public_scorecard_enabled=True,
        public_scorecard_path="public/scorecard/index.html",
        track_record_days=30,
    )


def _resolved_rows() -> list[dict[str, object]]:
    """Dwie zamknięte predykcje BULLISH +10% w jednym cyklu (dniu UTC)."""
    return [
        {
            "symbol": symbol,
            "predicted_trend": "BULLISH",
            "is_trend_correct": True,
            "price_at_prediction": "100",
            "actual_price_after_12h": "110",
            "timestamp": "2026-08-14T05:35:00+00:00",
        }
        for symbol in ("NVDA", "AMD")
    ]


def _repo(rows: list[dict[str, object]] | None = None) -> MagicMock:
    repository = MagicMock(spec=RepositoryPort)
    repository.get_resolved_predictions.return_value = _resolved_rows() if rows is None else rows
    repository.get_resolved_for_calibration.return_value = (
        [{"confidence_score": 0.8, "is_trend_correct": True}] if rows is None else []
    )
    repository.get_price_history.return_value = []
    return repository


def _wire(mocker: Any, repository: MagicMock) -> MagicMock:
    mocker.patch("main_agent.build_repository", return_value=repository)
    publisher = MagicMock(spec=WebPublisherPort)
    publisher.publish.return_value = "/runner/public/scorecard/index.html"
    mocker.patch("main_agent.build_public_scorecard_publisher", return_value=publisher)
    return publisher


def test_publishes_scorecard_from_repository_only(settings: Settings, mocker: Any) -> None:
    paid = [mocker.patch(f"main_agent.{name}") for name in PAID_BUILDERS]
    publisher = _wire(mocker, _repo())

    assert main_scorecard.main([], settings) == 0

    publisher.publish.assert_called_once()
    for builder in paid:
        builder.assert_not_called()


def test_published_page_is_the_net_of_costs_scorecard_without_symbols(
    settings: Settings, mocker: Any
) -> None:
    publisher = _wire(mocker, _repo())

    main_scorecard.main([], settings)

    page = publisher.publish.call_args.args[0]
    assert "paper-PnL (koszt 10 bps)" in page
    assert "NVDA" not in page and "AMD" not in page


def test_flag_off_exits_0_without_building_any_adapter(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="main_scorecard")
    build_repository = mocker.patch("main_agent.build_repository")
    build_publisher = mocker.patch("main_agent.build_public_scorecard_publisher")

    disabled = settings.model_copy(update={"public_scorecard_enabled": False})

    assert main_scorecard.main([], disabled) == 0

    build_repository.assert_not_called()
    build_publisher.assert_not_called()
    assert "public_scorecard_enabled" in caplog.text


def test_default_window_is_track_record_days(settings: Settings, mocker: Any) -> None:
    repository = _repo()
    _wire(mocker, repository)

    main_scorecard.main([], settings.model_copy(update={"track_record_days": 45}))

    # DAT-16: pokrycie danych czyta to samo okno drugi raz — każdy odczyt z oknem.
    assert repository.get_resolved_predictions.call_args_list
    assert {c.args for c in repository.get_resolved_predictions.call_args_list} == {(45,)}
    repository.get_resolved_for_calibration.assert_called_once_with(45)


def test_days_widens_every_read_and_the_disclosed_window(settings: Settings, mocker: Any) -> None:
    # Przy pauzie Fast Loopa 30 dni `track_record_days` daje pustą stronę (WIA-52).
    repository = _repo()
    publisher = _wire(mocker, repository)

    assert main_scorecard.main(["--days", "120"], settings) == 0

    # DAT-16: pokrycie danych czyta to samo okno drugi raz — każdy odczyt z oknem.
    assert repository.get_resolved_predictions.call_args_list
    assert {c.args for c in repository.get_resolved_predictions.call_args_list} == {(120,)}
    repository.get_resolved_for_calibration.assert_called_once_with(120)
    assert "ostatnie 120 dni" in publisher.publish.call_args.args[0]


@pytest.mark.parametrize("days", ["0", "-5", "abc"])
def test_days_must_be_a_positive_integer(settings: Settings, days: str) -> None:
    with pytest.raises(SystemExit) as error:
        main_scorecard.main(["--days", days], settings)

    assert error.value.code == 2


def test_empty_window_publishes_nothing_and_exits_0(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="main_scorecard")
    publisher = _wire(mocker, _repo(rows=[]))

    assert main_scorecard.main([], settings) == 0

    publisher.publish.assert_not_called()
    assert "brak artefaktu" in caplog.text


def test_repository_error_exits_1_so_the_workflow_alert_fires(
    settings: Settings, mocker: Any
) -> None:
    repository = _repo()
    repository.get_resolved_predictions.side_effect = RuntimeError("Supabase 503")
    publisher = _wire(mocker, repository)

    assert main_scorecard.main([], settings) == 1

    publisher.publish.assert_not_called()


def test_failed_write_exits_1(settings: Settings, mocker: Any) -> None:
    # `FileWebPublisher` połyka OSError i zwraca "" — bez pliku nie ma artefaktu Pages.
    publisher = _wire(mocker, _repo())
    publisher.publish.return_value = ""

    assert main_scorecard.main([], settings) == 1


# ----- workflow `scorecard_pages.yml` -----


def _spec() -> dict[Any, Any]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(spec, dict)
    return spec


def _steps() -> list[dict[str, Any]]:
    return [step for job in _spec()["jobs"].values() for step in job.get("steps", [])]


def _scorecard_step() -> dict[str, Any]:
    return next(s for s in _steps() if ENTRYPOINT in str(s.get("run", "")))


def test_workflow_runs_on_dispatch_with_the_weekly_cron_commented_out() -> None:
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = _spec()[True]

    assert "workflow_dispatch" in triggers
    assert "schedule" not in triggers, "cron zostaje zakomentowany na czas pauzy"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*#\s*schedule:", text, re.M), "brak zakomentowanego bloku schedule"
    assert re.search(r"^\s*#\s*-\s*cron:", text, re.M), "brak zakomentowanego crona"


def test_workflow_syncs_dependencies_without_extras() -> None:
    syncs = [str(s["run"]) for s in _steps() if "uv sync" in str(s.get("run", ""))]

    assert syncs
    assert not [cmd for cmd in syncs if "--extra" in cmd], syncs


def test_workflow_maps_only_the_free_secrets() -> None:
    env = {str(key): str(value) for key, value in (_scorecard_step().get("env") or {}).items()}

    assert {key for key, value in env.items() if "secrets." in value} == FREE_SECRET_KEYS
    # Settings wymaga pól płatnych kluczy do samej konstrukcji — zaślepki, nie sekrety.
    for key in ("OPENAI_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEYS"):
        assert env.get(key), f"brak zaślepki {key} w kroku scorecardu"
    assert not [key for key in PAID_SECRET_KEYS if "secrets." in env.get(key, "")]


def test_days_input_reaches_the_entrypoint_through_env_only() -> None:
    step = _scorecard_step()
    run = str(step["run"])
    env = step.get("env") or {}

    assert "--days" in run
    # Wejście dispatcha nigdy nie jest wklejane w skrypt (wstrzyknięcie powłoki).
    assert "${{" not in run
    assert any("inputs.days" in str(value) for value in env.values())


def test_every_job_ends_with_a_failure_alert() -> None:
    for job_name, job in _spec()["jobs"].items():
        last = job["steps"][-1]
        condition = str(last.get("if", ""))
        assert "failure()" in condition and "cancelled()" in condition, job_name
        assert "scripts/notify-failure.sh" in str(last.get("run", "")), job_name
        env = last.get("env", {})
        assert env.get("RESEND_API_KEY") == "${{ secrets.RESEND_API_KEY }}", job_name
        assert env.get("DIGEST_TO_EMAIL") == "${{ secrets.DIGEST_TO_EMAIL }}", job_name


def test_publishes_the_json_twin_next_to_the_page(settings: Settings, mocker: Any) -> None:
    # WIA-81: `scorecard.json` w katalogu strony — ten sam artefakt Pages.
    import json

    publisher = _wire(mocker, _repo())

    assert main_scorecard.main([], settings=settings) == 0

    name, content, content_type = publisher.publish_asset.call_args.args
    assert (name, content_type) == ("scorecard.json", "application/json")
    assert json.loads(content)["window_days"] == 30


def test_failed_json_write_exits_1(settings: Settings, mocker: Any) -> None:
    publisher = _wire(mocker, _repo())
    publisher.publish_asset.return_value = ""

    assert main_scorecard.main([], settings=settings) == 1


def _wire_status(mocker: Any, repository: MagicMock) -> MagicMock:
    """NIEZ-33: osobny publisher strony statusu (w katalogu scorecardu)."""
    from src.application.ports import CycleRun

    repository.get_cycle_runs_since.return_value = [
        CycleRun(
            loop="fast",
            run_id="1",
            run_attempt=1,
            started_at=datetime(2026, 10, 24, 5, 30, tzinfo=UTC),
            duration_s=10.0,
            eligible=42,
            saved=40,
            failures=2,
            paid_calls={"llm": 7},
            report_delivered=True,
        )
    ]
    status = MagicMock(spec=WebPublisherPort)
    status.publish.return_value = "/runner/public/scorecard/status/index.html"
    status.publish_asset.return_value = "/runner/public/scorecard/status/badge.json"
    mocker.patch("main_agent.build_public_status_publisher", return_value=status)
    return status


def test_status_page_is_published_with_the_scorecard(settings: Settings, mocker: Any) -> None:
    repository = _repo()
    _wire(mocker, repository)
    status = _wire_status(mocker, repository)
    settings.public_status_page_enabled = True

    assert main_scorecard.main([], settings=settings) == 0

    assert "Status StockAgenta" in status.publish.call_args.args[0]
    assert status.publish_asset.call_args.args[0] == "badge.json"


def test_status_page_flag_off_publishes_no_status(settings: Settings, mocker: Any) -> None:
    repository = _repo()
    _wire(mocker, repository)
    status = _wire_status(mocker, repository)

    assert main_scorecard.main([], settings=settings) == 0

    status.publish.assert_not_called()
    repository.get_cycle_runs_since.assert_not_called()


def test_unreadable_run_history_exits_1(settings: Settings, mocker: Any) -> None:
    from src.application.errors import ProviderUnavailable

    repository = _repo()
    _wire(mocker, repository)
    _wire_status(mocker, repository)
    repository.get_cycle_runs_since.side_effect = ProviderUnavailable("cycle_runs down")
    settings.public_status_page_enabled = True

    assert main_scorecard.main([], settings=settings) == 1


def test_failed_status_write_exits_1(settings: Settings, mocker: Any) -> None:
    repository = _repo()
    _wire(mocker, repository)
    status = _wire_status(mocker, repository)
    status.publish_asset.return_value = ""
    settings.public_status_page_enabled = True

    assert main_scorecard.main([], settings=settings) == 1
