# src/infrastructure/adapters/advisory_council.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Literal

from src.application.council_prompts import (
    chairman_prompt,
    investor_prompt,
)
from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import (
    CouncilInput,
    CouncilVerdict,
    InvestorOpinion,
    InvestorPersona,
    derive_consensus,
)

logger = logging.getLogger(__name__)

_VALID_RECS: frozenset[str] = frozenset({"BUY", "SELL", "HOLD"})
# Górna granica oczekiwania na CAŁY zestaw wywołań LLM równolegle.
# LLM port ma już własny timeout (30s) — ten jest buforem na sumę narzutu
# threadpool/scheduler. Po jego przekroczeniu kontynuujemy z opiniami,
# które zdążyły się zwrócić (graceful degradation).
_COUNCIL_TOTAL_TIMEOUT_S = 60.0

# Górny limit liczby person w radzie. Fan-out skaluje koszt LLM liniowo z
# liczbą plików person — bez capu dodanie pliku JSON niezauważalnie podbija
# rachunek za każdy cykl. 12 to praktyczny sufit: tyle różnych szkół
# inwestycyjnych w pełni pokrywa spektrum opinii, więcej to redundancja.
# (Follow-up: orchestrator mógłby wystawić to jako settings.council_max_personas.)
COUNCIL_MAX_PERSONAS = 12


def _parse_recommendation(raw: Any) -> Literal["BUY", "SELL", "HOLD"]:
    if isinstance(raw, str) and raw.upper() in _VALID_RECS:
        return raw.upper()  # type: ignore[return-value]
    return "HOLD"


def _clamp_unit(raw: Any, default: float = 0.5) -> float:
    """Przycina skalar dostarczony przez LLM do przedziału [0,1].

    Model dryfujący na skalę 0-100 (częsta usterka) zwracałby confidence=85,
    co bez bramki czyniłoby KAŻDĄ opinię HIGH (progi confidence_label to
    0.75/0.5) i korumpowało sygnał konwikcji (finding #20). Brak/None/błędny
    typ → wartość domyślna.
    """
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))


def _parse_opinion(name: str, raw: dict[str, Any]) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=_parse_recommendation(raw.get("recommendation")),
        confidence=_clamp_unit(raw.get("confidence")),
        reasoning=str(raw.get("reasoning") or ""),
        key_factors=tuple(str(f) for f in (raw.get("key_factors") or [])),
    )


class LLMAdvisoryCouncil(AdvisoryCouncilPort):
    """N równoległych wywołań LLM (jeden na inwestora) + wywołanie konsensusu.

    Personas wstrzykiwane przez konstruktor — żadne globalne źródło prawdy.
    Skład rady definiowany przez pliki JSON w `data/council_personas/`
    (parsowane przez `infrastructure.persona_loader`).
    """

    def __init__(
        self,
        llm_port: LLMPort,
        personas: tuple[InvestorPersona, ...],
        max_personas: int = COUNCIL_MAX_PERSONAS,
    ) -> None:
        if not personas:
            raise ValueError("LLMAdvisoryCouncil requires at least one persona.")
        self._llm = llm_port
        # Cap fan-outu: powyżej `max_personas` ucinamy radę i logujemy WARNING
        # z nazwami odrzuconych person. Kolejność person jest deterministyczna
        # (loader sortuje po name), więc ucinanie też jest stabilne.
        if len(personas) > max_personas:
            dropped = [p.name for p in personas[max_personas:]]
            logger.warning(
                "Advisory council: %d personas exceed cap of %d — "
                "truncating to %d, dropping: %s",
                len(personas),
                max_personas,
                max_personas,
                dropped,
            )
            personas = personas[:max_personas]
        self._personas = personas
        self._max_personas = max_personas

    def _call_investor(
        self, persona: InvestorPersona, data: CouncilInput
    ) -> InvestorOpinion:
        try:
            prompt = investor_prompt(persona, data)
            raw = self._llm.analyze(prompt)
            return _parse_opinion(persona.name, raw)
        except Exception:
            logger.exception(
                "Advisory council: %s failed — defaulting to HOLD", persona.name
            )
            return InvestorOpinion(
                investor_name=persona.name,
                recommendation="HOLD",
                confidence=0.0,
                reasoning="Błąd analizy.",
                key_factors=(),
            )

    def analyze(self, symbol: str, data: CouncilInput) -> CouncilVerdict:
        personas = self._personas
        opinions: list[InvestorOpinion | None] = [None] * len(personas)

        with ThreadPoolExecutor(max_workers=max(1, len(personas))) as executor:
            future_to_idx = {
                executor.submit(self._call_investor, persona, data): i
                for i, persona in enumerate(personas)
            }
            try:
                for future in as_completed(
                    future_to_idx, timeout=_COUNCIL_TOTAL_TIMEOUT_S
                ):
                    idx = future_to_idx[future]
                    opinions[idx] = future.result()
            except TimeoutError:
                pending = [
                    personas[i].name
                    for f, i in future_to_idx.items()
                    if not f.done()
                ]
                logger.warning(
                    "Advisory council: hit %ss total timeout — %d investor(s) pending: %s",
                    _COUNCIL_TOTAL_TIMEOUT_S,
                    len(pending),
                    pending,
                )
                # Anulujemy wiszące future'y (nie czekamy na nie w __exit__).
                for future in future_to_idx:
                    future.cancel()

        resolved_opinions: list[InvestorOpinion] = [
            op for op in opinions if op is not None
        ]

        try:
            prompt = chairman_prompt(resolved_opinions, data)
            raw = self._llm.analyze(prompt)
        except Exception:
            logger.exception("Advisory council: chairman call failed")
            raw = {}

        # AUTORYTATYWNA decyzja + siła konsensusu liczone z REALNYCH głosów
        # (domena), nie z liczby od chairmana. Dzięki temu jednogłośne BUY
        # zostaje BUY nawet gdy chairman padnie — nie ma cichego defaultu
        # HOLD/0.5 nadpisującego głosy inwestorów (finding #3).
        final_recommendation, consensus_strength = derive_consensus(resolved_opinions)

        # Chairman dostarcza już TYLKO warstwę narracyjną (summary, dissent).
        return CouncilVerdict(
            final_recommendation=final_recommendation,
            consensus_strength=consensus_strength,
            summary=str(raw.get("summary") or ""),
            dissenting_views=tuple(str(v) for v in (raw.get("dissenting_views") or [])),
            investor_opinions=tuple(resolved_opinions),
        )
