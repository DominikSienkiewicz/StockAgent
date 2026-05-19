# src/application/council_prompts.py
from __future__ import annotations

from src.domain.council import (
    CouncilInput,
    InvestorOpinion,
    InvestorPersona,
)
from src.domain.value_objects import ValuationVerdict


def _format_valuation_block(data: CouncilInput) -> str:
    """Formatuje blok wyceny do wstrzyknięcia w prompt inwestora.

    Zwraca pusty string gdy werdykt jest nieznany lub brak danych fundamentalnych.
    """
    if data.valuation_verdict is ValuationVerdict.UNKNOWN or data.fundamentals is None:
        return ""
    f = data.fundamentals

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    growth_pct = (
        f"{f.eps_growth_yoy * 100:.1f}%"
        if f.eps_growth_yoy is not None
        else "n/a"
    )
    return (
        f"\nValuation snapshot (as of {f.fetched_at.isoformat()}):\n"
        f"- Trailing P/E: {_fmt(f.trailing_pe)}\n"
        f"- Forward P/E: {_fmt(f.forward_pe)}\n"
        f"- PEG ratio: {_fmt(f.peg_ratio)}\n"
        f"- EPS growth YoY: {growth_pct}\n"
        f"- Deterministic verdict: {data.valuation_verdict.value}\n\n"
        "Consider this signal in your recommendation. You may disagree with "
        "the deterministic verdict if you justify it based on sector context "
        "or growth trajectory.\n"
    )


def investor_prompt(persona: InvestorPersona, data: CouncilInput) -> str:
    """Prompt dla pojedynczego inwestora w radzie.

    Układ cache-friendly: PRZEZ blokiem persony idą wszystkie dane wspólne
    (dane rynkowe, newsy, wycena, output_schema), które są identyczne dla
    15 wywołań w tym samym cyklu. Persona ląduje NA KOŃCU, żeby
    najdłuższy wspólny prefix mógł być cache'owany:
    - Anthropic ephemeral cache: ≥1024 tok prefixu, TTL 5 min — łapie się
      gdy prompt ma news + wycenę,
    - OpenAI automatyczny prompt cache: ≥1024 tok prefixu, też łapie.
    """
    news_formatted = "\n".join(f"  - {a}" for a in data.news_articles) or "  (brak newsów)"
    return f"""
<dane_rynkowe>
Symbol: {data.symbol}
Cena aktualna: {data.current_price}
Zmiana 12h: {data.price_delta_pct:+.2f}%
Sentyment (Alpha Vantage, skala -1.0 do +1.0): {data.sentiment_score:.2f}
Trend wg głównego analityka agenta: {data.llm_trend} (pewność: {data.llm_confidence:.0%})
Cel cenowy ML (XGBoost): {data.ml_price_target}
</dane_rynkowe>

<newsy>
{news_formatted}
</newsy>
{_format_valuation_block(data)}
<output_schema>
{{
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": float (0.0 - 1.0),
    "reasoning": "2-3 zdania uzasadnienia zgodnego z twoją filozofią inwestycyjną",
    "key_factors": ["czynnik 1", "czynnik 2", "czynnik 3"]
}}
</output_schema>

<rola>
Wcielasz się w {persona.name}. Twoja filozofia inwestycyjna:
{persona.style}
Analizujesz aktywo {data.symbol} powyżej i wydajesz rekomendację zgodną
z twoją filozofią. Odpowiadasz wyłącznie w formacie JSON wg powyższego
output_schema. Uzasadnienie po polsku.
</rola>
""".strip()


def chairman_prompt(opinions: list[InvestorOpinion], data: CouncilInput) -> str:
    opinions_formatted = "\n".join(
        f"  {op.investor_name}: {op.recommendation} "
        f"(pewność: {op.confidence:.0%}) — {op.reasoning}"
        for op in opinions
    )
    return f"""
<rola>
Jesteś przewodniczącym rady doradczej złożonej z {len(opinions)} legendarnych inwestorów.
Zebrałeś ich opinie na temat aktywa {data.symbol}. Twoim zadaniem jest synteza
tych opinii w finalną rekomendację rady. Bądź obiektywny — przedstaw zarówno
konsensus jak i istotne głosy niezgody.
Odpowiadasz wyłącznie w formacie JSON. Uzasadnienie po polsku.
</rola>

<opinie_rady>
{opinions_formatted}
</opinie_rady>

<dane_kontekstowe>
Symbol: {data.symbol}
Cena: {data.current_price}
Zmiana 12h: {data.price_delta_pct:+.2f}%
</dane_kontekstowe>

<output_schema>
{{
    "final_recommendation": "BUY" | "SELL" | "HOLD",
    "consensus_strength": float (0.0 - 1.0, gdzie 1.0 = pełna jednomyślność),
    "summary": "2-3 zdania syntezy stanowiska rady po polsku",
    "dissenting_views": ["Inwestor: powód niezgody", ...]
}}
</output_schema>
""".strip()
