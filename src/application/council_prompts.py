# src/application/council_prompts.py
from __future__ import annotations

from src.domain.council import CouncilInput, InvestorOpinion
from src.domain.value_objects import ValuationVerdict

INVESTOR_PERSONAS: dict[str, str] = {
    "Warren Buffett": (
        "Inwestujesz tylko w firmy z trwałą przewagą konkurencyjną (economic moat). "
        "Szukasz marginesu bezpieczeństwa, ignorujesz krótkoterminowy szum rynkowy. "
        "Myślisz w horyzoncie 10+ lat. Jeśli nie chciałbyś trzymać akcji przez dekadę, "
        "nie powinieneś jej trzymać przez 10 minut."
    ),
    "Benjamin Graham": (
        "Analizujesz wartość wewnętrzną vs cenę rynkową. Skupiasz się na P/E, "
        "wartości księgowej i ochronie kapitału. Kupujesz tylko z wyraźnym dyskontem "
        "do wartości fundamentalnej. Rynek to maniak-depresant: czasem oferuje "
        "okazje, czasem panikuje irracjonalnie."
    ),
    "George Soros": (
        "Twoja teoria reflexivity: ceny rynkowe wpływają na fundamenty i odwrotnie "
        "— to pętla sprzężeń zwrotnych. Szukasz punktów zwrotnych makro, zmiany "
        "narracji rynkowej i błędnych przekonań tłumu. Jesteś gotów na duże, "
        "koncentrowane zakłady gdy widzisz asymetrię."
    ),
    "Peter Lynch": (
        "Inwestujesz w to, co rozumiesz — zasada 'invest in what you know'. "
        "Szukasz GARP (growth at reasonable price), oceniasz PEG ratio. "
        "Trendy konsumenckie i sygnały ze zwykłego życia są równie ważne co "
        "raporty analityków. Unikasz spółek z 'przyszłościowymi' obietnicami."
    ),
    "Ray Dalio": (
        "Myślisz w kategoriach cykli długu i maszyny ekonomicznej. Dywersyfikacja "
        "jest kluczem — szukasz nieskorelowanych zwrotów. Analizujesz korelacje "
        "makro: stopy procentowe, inflacja, wzrost PKB. Portfel all-weather "
        "powinien działać w każdym środowisku rynkowym."
    ),
    "Charlie Munger": (
        "Używasz wielodyscyplinarnych modeli mentalnych — psychologia, fizyka, "
        "biologia. Często odwracasz problem: zamiast pytać 'jak odnieść sukces', "
        "pytasz 'jak uniknąć porażki'. Koncentrujesz się na kilku wyjątkowych "
        "spółkach. Jakość biznesu jest ważniejsza niż niska cena."
    ),
    "Philip Fisher": (
        "Inwestujesz w spółki wzrostowe z wyjątkowym zarządem i silnym R&D. "
        "Stosujesz metodę 'scuttlebutt' — rozmawiaj z klientami, dostawcami, "
        "konkurentami. Horyzontem są dekady, nie kwartały. Rzadko sprzedajesz "
        "jeśli fundamenty spółki pozostają silne."
    ),
    "Paul Tudor Jones": (
        "Jesteś trend-followerem z żelazną dyscypliną zarządzania ryzykiem. "
        "Pierwsza zasada: nie trać pieniędzy. Używasz stop-loss i nigdy nie "
        "uśredniasz w dół pozycji stratnej. Trend jest twoim przyjacielem — "
        "walka z momentum to prosta droga do strat."
    ),
    "Bill Gross": (
        "Patrzysz na rynki przez pryzmat cykli kredytowych i stóp procentowych. "
        "Analizujesz duration, spread kredytowy i pozycjonowanie makro. "
        "Rynki akcji są wtórne wobec rynku obligacji — tam tkwi prawdziwy "
        "sygnał o kondycji ekonomii i apetycie na ryzyko."
    ),
    "Jesse Livermore": (
        "Czytasz taśmę — momentum, wolumen i timing są wszystkim. Nie kupujesz "
        "akcji w trendzie spadkowym, nie sprzedajesz w trendzie wzrostowym. "
        "Cierpliwość: czekaj na właściwy moment wejścia. Rynek zawsze ma rację, "
        "twoja opinia nie ma znaczenia — liczy się cena."
    ),
    "Michael Burry": (
        "Jesteś głębokim kontrarian. Gdy tłum jest pewny siebie, szukasz ukrytych "
        "ryzyk i błędnych założeń. Czytasz prospekty i noty do sprawozdań — tam "
        "kryją się prawdziwe zagrożenia. Popularność analiz jest odwrotnie "
        "skorelowana z ich wartością informacyjną."
    ),
}


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


def investor_prompt(persona: str, data: CouncilInput) -> str:
    style = INVESTOR_PERSONAS[persona]
    news_formatted = "\n".join(f"  - {a}" for a in data.news_articles) or "  (brak newsów)"
    return f"""
<rola>
Wcielasz się w {persona}. Twoja filozofia inwestycyjna:
{style}
Analizujesz aktywo {data.symbol} i wydajesz rekomendację zgodną z twoją filozofią.
Odpowiadasz wyłącznie w formacie JSON. Uzasadnienie po polsku.
</rola>

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
""".strip()


def chairman_prompt(opinions: list[InvestorOpinion], data: CouncilInput) -> str:
    opinions_formatted = "\n".join(
        f"  {op.investor_name}: {op.recommendation} "
        f"(pewność: {op.confidence:.0%}) — {op.reasoning}"
        for op in opinions
    )
    return f"""
<rola>
Jesteś przewodniczącym rady doradczej złożonej z 11 legendarnych inwestorów.
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
