# src/infrastructure/adapters/advisory_council.py
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Literal

from src.application.council_prompts import (
    chairman_prompt,
    investor_prompt,
    investor_revision_prompt,
)
from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import (
    CouncilInput,
    CouncilVerdict,
    InvestorOpinion,
    InvestorPersona,
    derive_consensus,
)
from src.domain.value_objects import AssetType

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
        weights_provider: Callable[[AssetType], Mapping[str, float]] | None = None,
        debate_mode: bool = False,
    ) -> None:
        if not personas:
            raise ValueError("LLMAdvisoryCouncil requires at least one persona.")
        self._llm = llm_port
        # Feature #3: dostawca wag person z historycznej trafności (per typ aktywa).
        # None → ważenie wyłączone (każda persona waży 1.0, werdykt jak dotąd).
        self._weights_provider = weights_provider
        # Feature #1: jedna runda rewizji, gdy round-1 jest podzielone. Off → jak dziś.
        self._debate_mode = debate_mode
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

    def _hold_fallback(self, name: str) -> InvestorOpinion:
        return InvestorOpinion(
            investor_name=name,
            recommendation="HOLD",
            confidence=0.0,
            reasoning="Błąd analizy.",
            key_factors=(),
        )

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
            return self._hold_fallback(persona.name)

    def _call_revision(
        self,
        persona: InvestorPersona,
        data: CouncilInput,
        round1: list[InvestorOpinion],
    ) -> InvestorOpinion:
        # Pokazujemy personie stanowiska peerów (wszystkie poza nią) z rundy 1.
        peers = [op for op in round1 if op.investor_name != persona.name]
        try:
            prompt = investor_revision_prompt(persona, data, peers)
            raw = self._llm.analyze(prompt)
            return _parse_opinion(persona.name, raw)
        except Exception:
            logger.exception(
                "Advisory council: %s revision failed — keeping round-1 vote",
                persona.name,
            )
            # Degradacja: trzymamy głos z rundy 1, nie wymyślamy HOLD.
            for op in round1:
                if op.investor_name == persona.name:
                    return op
            return self._hold_fallback(persona.name)

    def _run_round(
        self,
        personas: tuple[InvestorPersona, ...],
        call: Callable[[InvestorPersona], InvestorOpinion],
    ) -> list[InvestorOpinion]:
        """Uruchamia jedną rundę wywołań person równolegle z budżetem czasowym.

        Współdzielone przez rundę 1 i rundę rewizji (debate mode) — ta sama
        maszyneria timeoutu/anulowania. Zwraca opinie, które zdążyły się zwrócić
        (graceful degradation: pojedynczy timeout nie wywala całego cyklu).
        """
        opinions: list[InvestorOpinion | None] = [None] * len(personas)
        with ThreadPoolExecutor(max_workers=max(1, len(personas))) as executor:
            future_to_idx = {
                executor.submit(call, persona): i
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
        return [op for op in opinions if op is not None]

    def _resolve_weights(self, data: CouncilInput) -> Mapping[str, float] | None:
        """Pobiera wagi person z providera (feature #3) lub None.

        Błąd providera nie może zatrzymać rady — degradujemy do braku wag
        (każda persona waży 1.0, werdykt jak bez ważenia).
        """
        if self._weights_provider is None:
            return None
        try:
            return self._weights_provider(data.asset_type)
        except Exception:
            logger.exception(
                "Advisory council: weights_provider failed — falling back to unweighted"
            )
            return None

    def _is_split(self, opinions: list[InvestorOpinion]) -> bool:
        # Split = jednocześnie BUY i SELL (pomijając HOLD) — fundamentalny brak
        # zgody co do kierunku. Reużywa semantyki CouncilVerdict.is_split_decision.
        recs = {op.recommendation for op in opinions}
        return "BUY" in recs and "SELL" in recs

    def analyze(self, symbol: str, data: CouncilInput) -> CouncilVerdict:
        personas = self._personas

        # --- Runda 1: równoległe opinie wszystkich person ---
        resolved_opinions = self._run_round(
            personas, lambda p: self._call_investor(p, data)
        )

        # --- Debate mode (feature #1): JEDNA runda rewizji, gdy round-1 split ---
        # Cap = dokładnie jedna runda rewizji; po niej derive_consensus z REVISED.
        if self._debate_mode and self._is_split(resolved_opinions):
            round1 = resolved_opinions
            resolved_opinions = self._run_round(
                personas, lambda p: self._call_revision(p, data, round1)
            )

        try:
            prompt = chairman_prompt(resolved_opinions, data)
            raw = self._llm.analyze(prompt)
        except Exception:
            logger.exception("Advisory council: chairman call failed")
            raw = {}

        # AUTORYTATYWNA decyzja + siła konsensusu liczone z REALNYCH głosów
        # (domena), nie z liczby od chairmana. Dzięki temu jednogłośne BUY
        # zostaje BUY nawet gdy chairman padnie — nie ma cichego defaultu
        # HOLD/0.5 nadpisującego głosy inwestorów (finding #3). Wagi (feature #3)
        # ważą głos każdej persony jej historyczną trafnością.
        weights = self._resolve_weights(data)
        final_recommendation, consensus_strength = derive_consensus(
            resolved_opinions, weights
        )

        # Chairman dostarcza już TYLKO warstwę narracyjną (summary, dissent).
        return CouncilVerdict(
            final_recommendation=final_recommendation,
            consensus_strength=consensus_strength,
            summary=str(raw.get("summary") or ""),
            dissenting_views=tuple(str(v) for v in (raw.get("dissenting_views") or [])),
            investor_opinions=tuple(resolved_opinions),
        )
