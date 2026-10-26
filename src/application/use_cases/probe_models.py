"""NIEZ-25: sonda dostępności skonfigurowanych modeli LLM (krok Slow Loopa).

Każdy model z konfiguracji (główny LLM, rada, router, zapasowy, embeddingi)
sprawdzany jest raz przez `ModelCatalogPort`. Brak modelu = CRITICAL ze źródłem
dostawcy (np. „Anthropic”) — przy wznowieniu crona po miesiącach pauzy wycofany
id to realne ryzyko. Błąd samej sondy (sieć, klucz bez dostępu do listy modeli)
to WARNING: nie wiadomo, czy model zniknął.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.application.ports import ModelCatalogPort
from src.domain.quota import QuotaAlert, QuotaSeverity

logger = logging.getLogger(__name__)

PROBE_ERROR_SOURCE = "Sonda modeli LLM"
_PROVIDER_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI"}


@dataclass(frozen=True)
class ModelProbe:
    """Model do sprawdzenia: dostawca (`anthropic` / `openai`), id i rola w agencie."""

    provider: str
    model_id: str
    role: str


def provider_label(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider)


class ProbeModelsUseCase:
    """`catalogs` — katalog per dostawca; dostawca bez katalogu (np. brak klucza)
    jest pomijany z WARNING-iem w logu, bez alertu."""

    def __init__(self, catalogs: Mapping[str, ModelCatalogPort]) -> None:
        self._catalogs = catalogs

    def run(self, probes: Sequence[ModelProbe], now: datetime) -> list[QuotaAlert]:
        roles: dict[tuple[str, str], list[str]] = {}
        for probe in probes:
            roles.setdefault((probe.provider, probe.model_id), []).append(probe.role)
        alerts: list[QuotaAlert] = []
        for (provider, model_id), model_roles in roles.items():
            alert = self._check(provider, model_id, ", ".join(model_roles), now)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def _check(self, provider: str, model_id: str, roles: str, now: datetime) -> QuotaAlert | None:
        catalog = self._catalogs.get(provider)
        if catalog is None:
            logger.warning("Model probe skipped for %s %s — no catalog", provider, model_id)
            return None
        try:
            available = catalog.is_available(model_id)
        except Exception as exc:
            # Sam typ wyjątku: komunikat SDK potrafi nieść URL albo fragment klucza.
            logger.warning(
                "Model probe failed for %s %s: %s", provider, model_id, type(exc).__name__
            )
            return QuotaAlert(
                source=PROBE_ERROR_SOURCE,
                severity=QuotaSeverity.WARNING,
                message=(
                    f"Nie sprawdzono modelu {model_id} ({roles}) w Models API "
                    f"{provider_label(provider)}: {type(exc).__name__}."
                ),
                action="Sprawdź sieć i zakres klucza API (dostęp do listy modeli).",
                occurred_at=now,
            )
        if available:
            logger.info("Model probe — %s %s available", provider, model_id)
            return None
        logger.error("Model probe — %s %s is not available", provider, model_id)
        return QuotaAlert(
            source=provider_label(provider),
            severity=QuotaSeverity.CRITICAL,
            message=(
                f"Model {model_id} ({roles}) nie jest dostępny w Models API — "
                "wycofany albo literówka w id; wywołania z nim padną."
            ),
            action=(
                "Zmień id modelu w config.toml (council_llm_model, router_*_model, "
                "llm_fallback_model) albo stałą DEFAULT_MODEL adaptera i wdroż przed cyklem."
            ),
            occurred_at=now,
        )
