"""NIEZ-25: adaptery Models API (Anthropic, OpenAI) i atrapa lokalna.

404 = model niedostępny (False); każdy inny błąd leci dalej — sonda zamienia go
na WARNING, bo błąd sieci to nie wycofany model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx
import openai
import pytest

from src.infrastructure.llm.model_catalog import (
    AnthropicModelCatalog,
    LocalStubModelCatalog,
    OpenAIModelCatalog,
)

_REQUEST = httpx.Request("GET", "https://api.example.com/v1/models/x")


def _not_found(module: object) -> Exception:
    error_type = module.NotFoundError  # type: ignore[attr-defined]
    return error_type(  # type: ignore[no-any-return]
        "model not found", response=httpx.Response(404, request=_REQUEST), body=None
    )


def test_anthropic_catalog_reports_an_existing_model(mocker: Any) -> None:
    client = MagicMock()
    mocker.patch("src.infrastructure.llm.model_catalog.Anthropic", return_value=client)

    assert AnthropicModelCatalog(api_key="sk-ant-test").is_available("claude-sonnet-4-6")
    client.models.retrieve.assert_called_once_with("claude-sonnet-4-6")


def test_anthropic_catalog_maps_404_to_unavailable(mocker: Any) -> None:
    client = MagicMock()
    client.models.retrieve.side_effect = _not_found(anthropic)
    mocker.patch("src.infrastructure.llm.model_catalog.Anthropic", return_value=client)

    assert AnthropicModelCatalog(api_key="sk-ant-test").is_available("claude-old") is False


def test_anthropic_catalog_lets_network_errors_through(mocker: Any) -> None:
    client = MagicMock()
    client.models.retrieve.side_effect = anthropic.APIConnectionError(request=_REQUEST)
    mocker.patch("src.infrastructure.llm.model_catalog.Anthropic", return_value=client)

    with pytest.raises(anthropic.APIConnectionError):
        AnthropicModelCatalog(api_key="sk-ant-test").is_available("claude-sonnet-4-6")


def test_openai_catalog_maps_404_to_unavailable(mocker: Any) -> None:
    client = MagicMock()
    client.models.retrieve.side_effect = _not_found(openai)
    mocker.patch("src.infrastructure.llm.model_catalog.OpenAI", return_value=client)

    assert OpenAIModelCatalog(api_key="sk-test").is_available("gpt-old") is False


def test_openai_catalog_reports_an_existing_model(mocker: Any) -> None:
    client = MagicMock()
    mocker.patch("src.infrastructure.llm.model_catalog.OpenAI", return_value=client)

    assert OpenAIModelCatalog(api_key="sk-test").is_available("gpt-5-mini")
    client.models.retrieve.assert_called_once_with("gpt-5-mini")


def test_clients_do_not_retry_on_their_own(mocker: Any) -> None:
    anthropic_cls = mocker.patch("src.infrastructure.llm.model_catalog.Anthropic")
    openai_cls = mocker.patch("src.infrastructure.llm.model_catalog.OpenAI")

    AnthropicModelCatalog(api_key="a")
    OpenAIModelCatalog(api_key="o")

    assert anthropic_cls.call_args.kwargs["max_retries"] == 0
    assert openai_cls.call_args.kwargs["max_retries"] == 0


def test_local_stub_says_every_model_is_available() -> None:
    assert LocalStubModelCatalog().is_available("anything")
