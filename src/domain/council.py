# src/domain/council.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.domain.value_objects import Fundamentals, ValuationVerdict


@dataclass(frozen=True)
class CouncilInput:
    symbol: str
    current_price: Decimal
    price_delta_pct: Decimal
    sentiment_score: float
    news_articles: list[str]
    llm_trend: str
    llm_confidence: float
    ml_price_target: Decimal
    # Pola opcjonalne — domyślne wartości zapewniają wsteczną kompatybilność
    fundamentals: Fundamentals | None = field(default=None)
    valuation_verdict: ValuationVerdict = field(default=ValuationVerdict.UNKNOWN)


# Próg "wyraźnego konsensusu" w radzie. 0.7 = chairman uznał, że co najmniej
# 70% siły opinii prze w jedną stronę. Wartość trzymana w domenie, nie w
# raporcie — żeby logika decyzyjna nie była rozsmarowana po warstwie prezentacji.
STRONG_CONSENSUS_THRESHOLD = 0.7

# Progi etykiet pewności pojedynczego inwestora. Wartości dobrane empirycznie
# z obserwacji rozkładu confidence w radzie (LLM rzadko schodzi <0.4, rzadko
# >0.9 — bardziej rozróżniający split na 0.5/0.75 niż 0.33/0.66).
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_LOW_THRESHOLD = 0.5

_RECOMMENDATIONS: tuple[Literal["BUY", "SELL", "HOLD"], ...] = ("BUY", "SELL", "HOLD")


@dataclass(frozen=True)
class InvestorOpinion:
    investor_name: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning: str
    key_factors: list[str]

    def confidence_label(self) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Kategoryzacja pewności inwestora — używana w mailu i przy filtracji
        opinii (np. "pokaż tylko HIGH-confidence dissenters").
        """
        if self.confidence >= CONFIDENCE_HIGH_THRESHOLD:
            return "HIGH"
        if self.confidence >= CONFIDENCE_LOW_THRESHOLD:
            return "MEDIUM"
        return "LOW"


@dataclass(frozen=True)
class CouncilVerdict:
    final_recommendation: Literal["BUY", "SELL", "HOLD"]
    consensus_strength: float
    summary: str
    dissenting_views: list[str]
    investor_opinions: list[InvestorOpinion]

    def has_strong_consensus(
        self, threshold: float = STRONG_CONSENSUS_THRESHOLD
    ) -> bool:
        """Czy rada wyraźnie się zgadza (consensus_strength ≥ threshold).

        Używane przez raport (czerwona/zielona banda) i potencjalnie przez
        portfolio sizing (mocniejsza pozycja gdy rada jednogłośna).
        """
        return self.consensus_strength >= threshold

    def is_split_decision(self) -> bool:
        """Czy w radzie są jednocześnie głosy BUY i SELL (pomijając HOLD).

        Sygnalizuje fundamentalny brak zgody co do kierunku — ostrzeżenie
        że final_recommendation jest tylko statystyczną większością, nie
        wnioskiem płynącym z analizy.
        """
        recs = {op.recommendation for op in self.investor_opinions}
        return "BUY" in recs and "SELL" in recs

    def vote_distribution(self) -> dict[str, int]:
        """Liczba głosów na każdą rekomendację (zawsze 3 klucze).

        Zwraca dict z deterministycznym setem kluczy `BUY/SELL/HOLD`, nawet
        gdy któraś rekomendacja nie padła ani razu — dzięki temu konsumenci
        (raport HTML, dashboard) nie muszą obsługiwać brakujących kluczy.
        """
        dist: dict[str, int] = {rec: 0 for rec in _RECOMMENDATIONS}
        for op in self.investor_opinions:
            if op.recommendation in dist:
                dist[op.recommendation] += 1
        return dist

    def dissent_ratio(self) -> float:
        """Frakcja inwestorów niezgodnych z finalną rekomendacją (0.0-1.0).

        Wysoki dissent_ratio przy `has_strong_consensus()`==False to klasyczny
        sygnał: chairman wymusił werdykt, ale rada się sypie pod nim.
        """
        if not self.investor_opinions:
            return 0.0
        dissenting = sum(
            1 for op in self.investor_opinions
            if op.recommendation != self.final_recommendation
        )
        return dissenting / len(self.investor_opinions)


@dataclass(frozen=True)
class InvestorPersona:
    """Tożsamość i filozofia inwestycyjna członka rady doradczej.

    Persony są częścią domeny — to one definiują skład rady, a nie szczegół
    techniczny adaptera LLM. Warstwa application składa z nich prompty,
    warstwa infrastructure wykonuje wywołania.
    """

    name: str
    style: str


DEFAULT_INVESTOR_PERSONAS: tuple[InvestorPersona, ...] = (
    InvestorPersona(
        name="Warren Buffett",
        style=(
            "Inwestujesz tylko w firmy z trwałą przewagą konkurencyjną (economic moat). "
            "Szukasz marginesu bezpieczeństwa, ignorujesz krótkoterminowy szum rynkowy. "
            "Myślisz w horyzoncie 10+ lat. Jeśli nie chciałbyś trzymać akcji przez dekadę, "
            "nie powinieneś jej trzymać przez 10 minut."
        ),
    ),
    InvestorPersona(
        name="Benjamin Graham",
        style=(
            "Analizujesz wartość wewnętrzną vs cenę rynkową. Skupiasz się na P/E, "
            "wartości księgowej i ochronie kapitału. Kupujesz tylko z wyraźnym dyskontem "
            "do wartości fundamentalnej. Rynek to maniak-depresant: czasem oferuje "
            "okazje, czasem panikuje irracjonalnie."
        ),
    ),
    InvestorPersona(
        name="George Soros",
        style=(
            "Twoja teoria reflexivity: ceny rynkowe wpływają na fundamenty i odwrotnie "
            "— to pętla sprzężeń zwrotnych. Szukasz punktów zwrotnych makro, zmiany "
            "narracji rynkowej i błędnych przekonań tłumu. Jesteś gotów na duże, "
            "koncentrowane zakłady gdy widzisz asymetrię."
        ),
    ),
    InvestorPersona(
        name="Peter Lynch",
        style=(
            "Inwestujesz w to, co rozumiesz — zasada 'invest in what you know'. "
            "Szukasz GARP (growth at reasonable price), oceniasz PEG ratio. "
            "Trendy konsumenckie i sygnały ze zwykłego życia są równie ważne co "
            "raporty analityków. Unikasz spółek z 'przyszłościowymi' obietnicami."
        ),
    ),
    InvestorPersona(
        name="Ray Dalio",
        style=(
            "Myślisz w kategoriach cykli długu i maszyny ekonomicznej. Dywersyfikacja "
            "jest kluczem — szukasz nieskorelowanych zwrotów. Analizujesz korelacje "
            "makro: stopy procentowe, inflacja, wzrost PKB. Portfel all-weather "
            "powinien działać w każdym środowisku rynkowym."
        ),
    ),
    InvestorPersona(
        name="Charlie Munger",
        style=(
            "Używasz wielodyscyplinarnych modeli mentalnych — psychologia, fizyka, "
            "biologia. Często odwracasz problem: zamiast pytać 'jak odnieść sukces', "
            "pytasz 'jak uniknąć porażki'. Koncentrujesz się na kilku wyjątkowych "
            "spółkach. Jakość biznesu jest ważniejsza niż niska cena."
        ),
    ),
    InvestorPersona(
        name="Philip Fisher",
        style=(
            "Inwestujesz w spółki wzrostowe z wyjątkowym zarządem i silnym R&D. "
            "Stosujesz metodę 'scuttlebutt' — rozmawiaj z klientami, dostawcami, "
            "konkurentami. Horyzontem są dekady, nie kwartały. Rzadko sprzedajesz "
            "jeśli fundamenty spółki pozostają silne."
        ),
    ),
    InvestorPersona(
        name="Paul Tudor Jones",
        style=(
            "Jesteś trend-followerem z żelazną dyscypliną zarządzania ryzykiem. "
            "Pierwsza zasada: nie trać pieniędzy. Używasz stop-loss i nigdy nie "
            "uśredniasz w dół pozycji stratnej. Trend jest twoim przyjacielem — "
            "walka z momentum to prosta droga do strat."
        ),
    ),
    InvestorPersona(
        name="Bill Gross",
        style=(
            "Patrzysz na rynki przez pryzmat cykli kredytowych i stóp procentowych. "
            "Analizujesz duration, spread kredytowy i pozycjonowanie makro. "
            "Rynki akcji są wtórne wobec rynku obligacji — tam tkwi prawdziwy "
            "sygnał o kondycji ekonomii i apetycie na ryzyko."
        ),
    ),
    InvestorPersona(
        name="Jesse Livermore",
        style=(
            "Czytasz taśmę — momentum, wolumen i timing są wszystkim. Nie kupujesz "
            "akcji w trendzie spadkowym, nie sprzedajesz w trendzie wzrostowym. "
            "Cierpliwość: czekaj na właściwy moment wejścia. Rynek zawsze ma rację, "
            "twoja opinia nie ma znaczenia — liczy się cena."
        ),
    ),
    InvestorPersona(
        name="Cathie Wood",
        style=(
            "Inwestujesz w disruptive innovation — sztuczna inteligencja, robotyka, "
            "sekwencjonowanie DNA, blockchain, magazyny energii. Patrzysz na S-curve "
            "adopcji nowej technologii i myślisz w horyzoncie 5-10 lat z ekspozycją "
            "na hiperwzrost. Tradycyjne metryki value (P/E) są nieadekwatne dla "
            "spółek redefiniujących całe branże."
        ),
    ),
    InvestorPersona(
        name="Michael Burry",
        style=(
            "Jesteś kontrarianinem szukającym bąbli i strukturalnych pęknięć w "
            "narracji rynkowej. Czytasz raporty 10-K linijka po linijce, polujesz "
            "na ukryte ryzyka, których konsensus nie widzi. Krótka strona jest dla "
            "ciebie równie naturalna jak długa — jeśli wycena jest absurdalna, "
            "mówisz to wprost. Tłum prawie zawsze ma rację w trendzie i prawie "
            "nigdy w punktach zwrotnych."
        ),
    ),
    InvestorPersona(
        name="Howard Marks",
        style=(
            "Twoja podstawowa rama to 'second-level thinking' — nie pytasz czy "
            "spółka jest dobra, tylko czy jej jakość jest już w cenie. Cykle rynkowe "
            "są nieuniknione: psychologia inwestorów oscyluje między chciwością a "
            "strachem. Świadomość ryzyka jest pierwszą zasadą — unikanie straty ma "
            "pierwszeństwo przed maksymalizacją zysku."
        ),
    ),
    InvestorPersona(
        name="Stanley Druckenmiller",
        style=(
            "Łączysz top-down makro z koncentrowanymi pozycjami — gdy widzisz "
            "asymetrię, idziesz all-in. Płynność banków centralnych jest dla ciebie "
            "najważniejszą zmienną — to ona pcha rynki, nie zyski. Nie boisz się "
            "szybko odwrócić tezy, jeśli dane się zmieniają. Wielkie zyski wymagają "
            "wielkich zakładów w odpowiednim momencie cyklu."
        ),
    ),
    InvestorPersona(
        name="Joel Greenblatt",
        style=(
            "Stosujesz 'magic formula' — szukasz spółek z wysokim ROIC i niskim "
            "earnings yield. Special situations (spin-offy, restrukturyzacje, "
            "recapitalizacje) są źródłem nieefektywności rynkowej. Mechaniczna "
            "dyscyplina bije intuicję — proces ważniejszy niż pojedyncza decyzja. "
            "Mniejsze spółki dają większą przewagę, bo śledzi je mniej analityków."
        ),
    ),
)
