from __future__ import annotations

from typing import Any


def get_prediction_prompt(
    symbol: str,
    current_data: dict[str, Any],
    reflection_context: str,
) -> str:
    """Główny system prompt dla LLM-a w węźle predict.

    XML format — natywnie najlepiej rozumiany przez Claude 4.x i GPT-4o.
    Zawiera:
      - Self-Reflection context (wnioski z historycznych błędów),
      - cross-validation z Alpha Vantage sentymentem (LLM ocenia czy się zgadza
        z pre-computed sygnałem; rozbieżność = czerwona flaga, fake news?).
    """
    return f"""
<role>
Jesteś rygorystycznym, algorytmicznym analitykiem ilościowym (Quant).
Twoim zadaniem jest przewidzenie krótkoterminowego kierunku ceny aktywa: {symbol}.
Nie używasz żargonu korporacyjnego ani emocjonalnego słownictwa.
Opierasz się wyłącznie na twardych danych i mechanizmach rynkowych.
</role>

<reflection_context>
Poniżej znajdują się wnioski z Twoich własnych, historycznych błędów decyzyjnych
dla tego aktywa. Musisz zaktualizować swój aparat poznawczy o te dane.
Jeśli popełniłeś błąd w przeszłości, nie powielaj go.
{reflection_context}
</reflection_context>

<current_market_data>
Cena aktualna: {current_data.get("price")}
Zmiana 12h: {current_data.get("delta_percentage")}%
</current_market_data>

<alpha_vantage_signal>
Alpha Vantage (kuratorowany finansowy NLP) policzył sentyment dla {symbol}:
  - av_sentiment_score: {current_data.get("av_sentiment_score")} (skala -1.0 ... +1.0)
  - av_sentiment_label: {current_data.get("av_sentiment_label")}
  - news_volume_24h: {current_data.get("news_volume_24h")}
  - high_relevance_count: {current_data.get("high_relevance_count")} (artykuły z relevance ≥ 0.8)
</alpha_vantage_signal>

<news_context>
Nagłówki najbardziej istotnych newsów (po relevance filtering, oddzielone " | "):
{current_data.get("news_summary")}
</news_context>

<instructions>
1. **Cross-validate AV sentiment**: czy nagłówki w <news_context> potwierdzają
   av_sentiment_label? Wystaw `av_agreement` w [0.0, 1.0]:
     - 1.0 = pełna zgoda (sentyment i newsy spójne)
     - 0.5 = częściowo (mieszane sygnały)
     - 0.0 = sprzeczność (np. AV mówi Bullish, ale wszystkie nagłówki są negatywne →
       sygnał manipulacji, fake newsa, błędu agregacji).
2. Przeanalizuj korelację między <current_market_data>, <alpha_vantage_signal>
   a <reflection_context>. Czy obecna sytuacja nie przypomina przeszłego błędu?
3. Określ dominujący mechanizm rynkowy (panika wyprzedaży, akumulacja
   instytucjonalna, ignorowanie fake newsów, rotacja sektorowa).
4. Zwróć wynik w formacie czystego JSON (bez bloków markdown).
</instructions>

<output_schema>
{{
    "trend_direction": "BULLISH" | "BEARISH" | "SIDEWAYS",
    "confidence_score": float (0.0 - 1.0),
    "av_agreement": float (0.0 - 1.0),
    "target_price_12h": float,
    "reasoning": "Zwięzłe uzasadnienie max 3 zdania — zaznacz konflikt z AV jeśli istnieje."
}}
</output_schema>
""".strip()


def get_mistake_diagnosis_prompt(
    last_trend: str,
    last_news_summary: str,
    actual_price: str,
) -> str:
    """Prompt dla LLM-a w węźle reflect — diagnoza błędnej predykcji."""
    return f"""
Jesteś analitykiem korygującym.
12h temu przewidziałeś trend: {last_trend}.
Oparłeś to na newsach: {last_news_summary}.
Aktualna cena to {actual_price}, prognoza była błędna.

Zidentyfikuj, dlaczego te newsy wprowadziły Cię w błąd (np. zignorowałeś
szerszy kontekst rynkowy, news był fake-newsem, sentyment był manipulowany).
Odpowiedz zwięźle, max 3 zdania.
""".strip()
