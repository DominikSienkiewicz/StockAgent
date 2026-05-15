# src/infrastructure/adapters/advisory_council.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from src.application.council_prompts import (
    INVESTOR_PERSONAS,
    chairman_prompt,
    investor_prompt,
)
from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import CouncilInput, CouncilVerdict, InvestorOpinion

logger = logging.getLogger(__name__)

_VALID_RECS: frozenset[str] = frozenset({"BUY", "SELL", "HOLD"})


def _parse_recommendation(raw: Any) -> Literal["BUY", "SELL", "HOLD"]:
    if isinstance(raw, str) and raw.upper() in _VALID_RECS:
        return raw.upper()  # type: ignore[return-value]
    return "HOLD"


def _parse_opinion(name: str, raw: dict[str, Any]) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=_parse_recommendation(raw.get("recommendation")),
        confidence=float(raw.get("confidence") or 0.5),
        reasoning=str(raw.get("reasoning") or ""),
        key_factors=list(raw.get("key_factors") or []),
    )


class LLMAdvisoryCouncil(AdvisoryCouncilPort):
    """11 równoległych wywołań LLM (jeden na inwestora) + wywołanie konsensusu."""

    def __init__(self, llm_port: LLMPort) -> None:
        self._llm = llm_port

    def _call_investor(self, name: str, data: CouncilInput) -> InvestorOpinion:
        try:
            prompt = investor_prompt(name, data)
            raw = self._llm.analyze(prompt)
            return _parse_opinion(name, raw)
        except Exception:
            logger.exception("Advisory council: %s failed — defaulting to HOLD", name)
            return InvestorOpinion(
                investor_name=name,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Błąd analizy.",
                key_factors=[],
            )

    def analyze(self, symbol: str, data: CouncilInput) -> CouncilVerdict:
        investor_names = list(INVESTOR_PERSONAS.keys())
        opinions: list[InvestorOpinion | None] = [None] * len(investor_names)

        with ThreadPoolExecutor(max_workers=11) as executor:
            future_to_idx = {
                executor.submit(self._call_investor, name, data): i
                for i, name in enumerate(investor_names)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                opinions[idx] = future.result()

        resolved_opinions: list[InvestorOpinion] = [
            op for op in opinions if op is not None
        ]

        try:
            prompt = chairman_prompt(resolved_opinions, data)
            raw = self._llm.analyze(prompt)
        except Exception:
            logger.exception("Advisory council: chairman call failed")
            raw = {}

        return CouncilVerdict(
            final_recommendation=_parse_recommendation(raw.get("final_recommendation")),
            consensus_strength=float(raw.get("consensus_strength") or 0.5),
            summary=str(raw.get("summary") or ""),
            dissenting_views=list(raw.get("dissenting_views") or []),
            investor_opinions=resolved_opinions,
        )
