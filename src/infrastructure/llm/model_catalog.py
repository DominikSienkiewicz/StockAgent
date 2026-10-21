"""NIEZ-25: adaptery `ModelCatalogPort` — Models API Anthropic i OpenAI.

`models.retrieve` nie zużywa tokenów. 404 (`NotFoundError`) znaczy „dostawca nie
zna modelu”; każdy inny błąd leci dalej, bo błąd sieci to nie wycofany model.
Klienci bez własnych ponowień: sonda chodzi raz w tygodniu, a jej błąd i tak
kończy się WARNING-iem.
"""

from __future__ import annotations

from anthropic import Anthropic
from anthropic import NotFoundError as AnthropicNotFoundError
from openai import NotFoundError as OpenAINotFoundError
from openai import OpenAI

from src.application.ports import ModelCatalogPort

DEFAULT_TIMEOUT = 15.0


class AnthropicModelCatalog(ModelCatalogPort):
    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._client = Anthropic(api_key=api_key, timeout=timeout, max_retries=0)

    def is_available(self, model_id: str) -> bool:
        try:
            self._client.models.retrieve(model_id)
        except AnthropicNotFoundError:
            return False
        return True


class OpenAIModelCatalog(ModelCatalogPort):
    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def is_available(self, model_id: str) -> bool:
        try:
            self._client.models.retrieve(model_id)
        except OpenAINotFoundError:
            return False
        return True


class LocalStubModelCatalog(ModelCatalogPort):
    """FIN-35: lokalnie bez płatnych portów — sonda nie wysyła requestu."""

    def is_available(self, model_id: str) -> bool:
        return True
