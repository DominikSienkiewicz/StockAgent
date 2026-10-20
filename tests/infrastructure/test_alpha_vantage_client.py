"""Testy `AlphaVantageClient` — wspólny klient dla News i Sentiment adapterów."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
import requests

from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient


def _ok_response(payload: dict) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


_SAMPLE_PAYLOAD = {
    "items": "3",
    "sentiment_score_definition": "...",
    "feed": [
        {
            "title": "AMD surges on AI chip demand",
            "url": "https://example.com/1",
            "time_published": "20260514T093000",
            "summary": "...",
            "source": "Bloomberg",
            "overall_sentiment_score": 0.45,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "AMD",
                    "relevance_score": "0.95",
                    "ticker_sentiment_score": "0.52",
                    "ticker_sentiment_label": "Bullish",
                },
                {
                    "ticker": "NVDA",
                    "relevance_score": "0.31",
                    "ticker_sentiment_score": "0.21",
                    "ticker_sentiment_label": "Neutral",
                },
            ],
        },
        {
            "title": "Apple Q2 beats expectations",
            "url": "https://example.com/2",
            "time_published": "20260514T080000",
            "summary": "...",
            "source": "Reuters",
            "overall_sentiment_score": 0.61,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.92",
                    "ticker_sentiment_score": "0.65",
                    "ticker_sentiment_label": "Bullish",
                },
            ],
        },
        {
            "title": "Random tech mention",
            "url": "https://example.com/3",
            "time_published": "20260513T220000",
            "summary": "...",
            "source": "Some Blog",
            "overall_sentiment_score": -0.05,
            "overall_sentiment_label": "Neutral",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.10",  # niski relevance — powinien być odfiltrowany
                    "ticker_sentiment_score": "0.05",
                    "ticker_sentiment_label": "Neutral",
                },
            ],
        },
    ],
}


@pytest.fixture
def client() -> AlphaVantageClient:
    return AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL", "AMD", "NVDA"])


class TestPerTickerFetch:
    """Domyślnie batch_size=1 — AV `tickers` to AND filter, heterogeniczna lista
    zwraca pusty feed. Robimy osobny request per symbol."""

    def test_makes_one_request_per_symbol(self, client, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        client.articles_for("AAPL")
        client.articles_for("AMD")
        client.articles_for("NVDA")
        client.sentiment_for("AAPL")

        # 3 symbole × 1 request każdy = 3 requesty (potem cache)
        assert mock_get.call_count == 3
        # Każdy request niesie pojedynczy ticker
        tickers_sent = [
            call.kwargs["params"]["tickers"] for call in mock_get.call_args_list
        ]
        assert tickers_sent == ["AAPL", "AMD", "NVDA"]

    def test_caches_after_first_full_fetch(self, client, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        # 4 wywołania, ale tylko 3 unikalne symbole (AAPL, AMD, NVDA)
        client.articles_for("AAPL")
        client.sentiment_for("AMD")
        client.articles_for("NVDA")
        client.sentiment_for("AAPL")   # cache

        assert mock_get.call_count == 3   # raz na symbol, nie 4

    def test_passes_apikey_in_params(self, client, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )
        client.articles_for("AAPL")
        params = mock_get.call_args.kwargs["params"]
        assert params["function"] == "NEWS_SENTIMENT"
        assert params["apikey"] == "av-test"


class TestBatching:
    def test_splits_large_symbol_lists_into_batches(self, mocker):
        # 22 symbole, batch_size=10 → 3 batche (10+10+2)
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )
        # Sleep mock — nie chcemy 2× 1.1s w teście
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")

        symbols = [f"SYM{i:02d}" for i in range(22)]
        client = AlphaVantageClient(api_keys=["k"], symbols=symbols, batch_size=10)
        client.articles_for("SYM00")

        assert mock_get.call_count == 3
        batches = [
            call.kwargs["params"]["tickers"].split(",")
            for call in mock_get.call_args_list
        ]
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 2

    def test_sleeps_between_batches(self, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )
        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )

        symbols = [f"SYM{i:02d}" for i in range(15)]
        client = AlphaVantageClient(api_keys=["k"], symbols=symbols, batch_size=10)
        client.articles_for("SYM00")

        # 2 batche → 1 sleep między nimi
        assert sleep_mock.call_count == 1
        assert sleep_mock.call_args.args[0] >= 1.0

    def test_deduplicates_articles_by_url_across_batches(self, mocker):
        # Jeden artykuł pojawia się w dwóch batchach (np. wzmianka o AAPL i MSFT)
        duplicated_article = _SAMPLE_PAYLOAD["feed"][0]
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"feed": [duplicated_article]}),
        )
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")

        symbols = [f"SYM{i:02d}" for i in range(15)]
        client = AlphaVantageClient(api_keys=["k"], symbols=symbols, batch_size=10)
        client._fetch()

        assert len(client._cached_feed or []) == 1

    def test_handles_rate_limit_info_response_gracefully(self, mocker, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        # AV przy rate-limit zwraca 200 + Information, bez feed
        rate_limit_response = _ok_response({
            "Information": (
                "Thank you for using Alpha Vantage! "
                "Please consider spreading out your free API requests..."
            )
        })
        mocker.patch(
            "requests.Session.get",
            return_value=rate_limit_response,
        )
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")

        client = AlphaVantageClient(api_keys=["k"], symbols=["AAPL"])
        result = client._fetch()

        # Nie raise — graceful degradation: pusty feed
        assert result == []
        assert "rate-limit" in caplog.text.lower()


class TestApiKeyRotation:
    """Wiele kluczy → klient rotuje gdy AV zwraca rate-limit info."""

    def _rate_limit_response(self):
        return _ok_response({"Information": "rate limit reached for key X"})

    def test_rotates_to_next_key_when_first_is_rate_limited(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get"
        )
        # AAPL: rate-limit z key1 → rotacja → sukces z key2.
        # AMD/NVDA: sukces z key2 (już aktywny).
        mock_get.side_effect = [
            self._rate_limit_response(),       # AAPL / key1 fail
            _ok_response(_SAMPLE_PAYLOAD),     # AAPL / key2 OK
            _ok_response(_SAMPLE_PAYLOAD),     # AMD  / key2 OK
            _ok_response(_SAMPLE_PAYLOAD),     # NVDA / key2 OK
        ]

        client = AlphaVantageClient(
            api_keys=["key1", "key2"], symbols=["AAPL", "AMD", "NVDA"]
        )
        client._fetch()

        # 4 wywołania REST (1 fail + 3 success)
        assert mock_get.call_count == 4
        keys_used = [
            call.kwargs["params"]["apikey"] for call in mock_get.call_args_list
        ]
        # Pierwszy z key1 (fail), wszystkie pozostałe z key2 (aktywny po rotacji)
        assert keys_used == ["key1", "key2", "key2", "key2"]
        assert client._active_key_idx == 1

    def test_keeps_active_key_across_subsequent_batches(self, mocker):
        # Gdy key#1 zadziała w pierwszym batchu, nie wracamy do key#0
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get"
        )
        mock_get.side_effect = [
            self._rate_limit_response(),       # batch 1 z key1 — fail
            _ok_response(_SAMPLE_PAYLOAD),     # batch 1 z key2 — OK
            _ok_response(_SAMPLE_PAYLOAD),     # batch 2 z key2 — OK (bez powrotu do key1)
        ]
        symbols = [f"SYM{i:02d}" for i in range(15)]   # 2 batche
        client = AlphaVantageClient(
            api_keys=["key1", "key2"], symbols=symbols, batch_size=10
        )
        client._fetch()

        keys_used = [
            call.kwargs["params"]["apikey"] for call in mock_get.call_args_list
        ]
        assert keys_used == ["key1", "key2", "key2"]

    def test_returns_partial_feed_when_all_keys_exhausted(self, mocker, caplog):
        import logging
        caplog.set_level(logging.ERROR)
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get"
        )
        # Wszystkie 3 klucze rate-limited
        mock_get.side_effect = [self._rate_limit_response()] * 3

        client = AlphaVantageClient(
            api_keys=["k1", "k2", "k3"], symbols=["AAPL"]
        )
        result = client._fetch()

        assert result == []
        assert mock_get.call_count == 3
        assert "exhausted" in caplog.text.lower()

    def test_exposes_degraded_reason_when_all_keys_exhausted(self, mocker):
        # Sygnał dla warstwy aplikacji: bieżący cykl działał na (potencjalnie)
        # niekompletnym feedzie. Bez tego sygnału newsy=[] dla wszystkich
        # symboli wyglądają identycznie jak "spokojny dzień, brak newsów".
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch("requests.Session.get")
        mock_get.side_effect = [self._rate_limit_response()] * 2

        client = AlphaVantageClient(api_keys=["k1", "k2"], symbols=["AAPL"])
        client._fetch()

        assert client.degraded_reason == "av_keys_exhausted"

    def test_degraded_reason_is_none_when_first_call_succeeds(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )
        client = AlphaVantageClient(api_keys=["k1", "k2"], symbols=["AAPL"])
        client._fetch()

        assert client.degraded_reason is None

    def test_partial_feed_when_keys_exhaust_mid_fetch(self, mocker):
        # 2 batche, 2 klucze. Batch 1 z key1 → OK. Batch 2: key1 rate-limit,
        # key2 też rate-limit → klucze wyczerpane, ale batch 1 ma już dane.
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get"
        )
        mock_get.side_effect = [
            _ok_response(_SAMPLE_PAYLOAD),     # batch 1 / key1 OK
            self._rate_limit_response(),       # batch 2 / key1 fail
            self._rate_limit_response(),       # batch 2 / key2 fail
        ]
        symbols = [f"SYM{i:02d}" for i in range(15)]
        client = AlphaVantageClient(
            api_keys=["k1", "k2"], symbols=symbols, batch_size=10
        )
        client._fetch()

        # Mamy częściowy feed z batcha 1
        assert client._cached_feed is not None
        assert len(client._cached_feed) > 0

    @staticmethod
    def _note_response():
        # Historyczny kształt limitu częstotliwości AV: pole `Note` zamiast `Information`.
        return _ok_response({
            "Note": "Thank you for using Alpha Vantage! Our standard API call "
            "frequency is 5 calls per minute."
        })

    def test_rotates_when_soft_limit_uses_note_field(self, mocker):
        # DAT-06: `Note` to też soft-limit. Bez wspólnego klasyfikatora klient brał
        # go za pusty feed — bez rotacji i bez `degraded_reason`.
        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mock_get = mocker.patch("requests.Session.get")
        mock_get.side_effect = [
            self._note_response(),             # key1 — limit minutowy
            self._note_response(),             # key1 po backoffie — nadal limit
            _ok_response(_SAMPLE_PAYLOAD),     # key2 — OK
        ]

        client = AlphaVantageClient(api_keys=["key1", "key2"], symbols=["AAPL"])
        articles = client.articles_for("AAPL")

        keys_used = [call.kwargs["params"]["apikey"] for call in mock_get.call_args_list]
        # Krótki backoff i ponowienie TYM SAMYM kluczem, dopiero potem rotacja.
        assert keys_used == ["key1", "key1", "key2"]
        assert sleep_mock.call_count == 1
        assert sleep_mock.call_args.args[0] > 0
        assert len(articles) == 1
        assert client.degraded_reason is None

    def test_note_cleared_by_backoff_keeps_the_same_key(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch("requests.Session.get")
        mock_get.side_effect = [self._note_response(), _ok_response(_SAMPLE_PAYLOAD)]

        client = AlphaVantageClient(api_keys=["key1", "key2"], symbols=["AAPL"])
        articles = client.articles_for("AAPL")

        keys_used = [call.kwargs["params"]["apikey"] for call in mock_get.call_args_list]
        assert keys_used == ["key1", "key1"]
        assert client._active_key_idx == 0
        assert len(articles) == 1

    def test_note_on_every_key_alerts_about_the_frequency_limit(self, mocker):
        # Alert „dzienny limit 25 zapytań” myliłby, gdy klucze odbił limit minutowy.
        from src.application.quota_monitor import QuotaMonitor

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get", return_value=self._note_response()
        )
        monitor = QuotaMonitor()
        client = AlphaVantageClient(
            api_keys=["key1", "key2"], symbols=["AAPL"], quota_monitor=monitor
        )

        client.articles_for("AAPL")

        # Dwa klucze × (próba + ponowienie po backoffie).
        assert mock_get.call_count == 4
        assert client.degraded_reason == "av_keys_exhausted"
        alert = monitor.alerts[0]
        assert "częstotliwości" in alert.message
        assert "dzienny limit" not in alert.message
        assert "ALPHA_VANTAGE_API_KEYS" in alert.action


class TestArticlesFor:
    def test_filters_by_symbol_via_ticker_sentiment(self, client, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        amd_articles = client.articles_for("AMD")

        assert len(amd_articles) == 1
        assert amd_articles[0]["title"] == "AMD surges on AI chip demand"

    def test_filters_by_relevance_threshold(self, client, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        # AAPL ma 2 wzmianki: relevance 0.92 i 0.10
        # Domyślny próg 0.5 → tylko ta z 0.92
        aapl_articles = client.articles_for("AAPL")

        assert len(aapl_articles) == 1
        assert aapl_articles[0]["title"] == "Apple Q2 beats expectations"

    def test_custom_relevance_threshold(self, mocker):
        client = AlphaVantageClient(
            api_keys=["av-test"],
            symbols=["AAPL"],
            relevance_threshold=0.05,
        )
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        articles = client.articles_for("AAPL")
        assert len(articles) == 2  # oba przechodzą

    def test_articles_include_normalized_fields(self, client, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        articles = client.articles_for("AMD")
        article = articles[0]

        assert article["title"] == "AMD surges on AI chip demand"
        assert article["source"] == "Bloomberg"
        assert article["url"] == "https://example.com/1"
        assert article["published_at"] == "20260514T093000"
        # Per-ticker fields
        assert article["relevance_score"] == pytest.approx(0.95)
        assert article["ticker_sentiment_score"] == pytest.approx(0.52)


class TestSentimentFor:
    def test_computes_relevance_weighted_average(self, client, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )

        result = client.sentiment_for("AAPL")

        # Tylko 1 artykuł z relevance >= 0.5 — sentyment = 0.65
        assert result["av_sentiment_score"] == pytest.approx(0.65, abs=1e-4)
        assert result["av_relevance_avg"] == pytest.approx(0.92, abs=1e-4)
        assert result["news_volume_24h"] == 1
        assert result["high_relevance_count"] == 1   # > 0.8 (jeden z relevance 0.92)
        assert result["av_sentiment_label"] in {"Bullish", "Somewhat-Bullish", "Neutral"}

    def test_returns_neutral_defaults_when_no_articles(self, client, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"feed": []}),
        )

        result = client.sentiment_for("AAPL")

        assert result["av_sentiment_score"] == pytest.approx(0.0)
        assert result["news_volume_24h"] == 0
        assert result["av_sentiment_label"] == "Neutral"

    def test_handles_missing_feed_field(self, client, mocker):
        # AV przy nadmiernym rate-limit czasem zwraca {"Information": "..."}
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"Information": "rate limit reached"}),
        )

        result = client.sentiment_for("AAPL")
        assert result["news_volume_24h"] == 0


class TestSymbolFiltering:
    def test_filters_out_tickers_with_dot_notation(self, caplog, mocker):
        # CSPX.L (LSE) zawiera kropkę — AV odrzuciłby cały batch.
        import logging
        caplog.set_level(logging.WARNING)
        client = AlphaVantageClient(api_keys=["k"], symbols=["AAPL", "CSPX.L", "AMD"])
        assert client._symbols == ["AAPL", "AMD"]
        assert "CSPX.L" in caplog.text

    def test_keeps_dash_underscore_colon(self):
        client = AlphaVantageClient(api_keys=["k"], symbols=["BRK-A", "RDS_A", "FOO:BAR"])
        assert client._symbols == ["BRK-A", "RDS_A", "FOO:BAR"]


class TestErrorHandling:
    def test_raises_on_http_error(self, client, mocker):
        response = MagicMock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mocker.patch(
            "requests.Session.get",
            return_value=response,
        )

        with pytest.raises(requests.HTTPError):
            client.articles_for("AAPL")

    def test_mid_fetch_failure_caches_partial_feed_and_is_not_refetched(
        self, mocker, caplog
    ):
        # DAT-08: jeden padnięty batch nie może wymuszać pełnego ponownego
        # pobrania przy każdym kolejnym symbolu — częściowy feed zostaje w cache.
        import logging

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_SAMPLE_PAYLOAD),                       # AAPL — OK
                requests.ConnectionError(
                    "Max retries exceeded with url: /query?apikey=av-secret-key"
                ),                                                   # AMD — sieć padła
            ],
        )
        client = AlphaVantageClient(
            api_keys=["av-secret-key"], symbols=["AAPL", "AMD", "NVDA"]
        )

        # Wariant zgodny wstecz: pierwszy caller nadal dostaje wyjątek.
        with caplog.at_level(logging.WARNING), pytest.raises(requests.ConnectionError):
            client.articles_for("AAPL")
        aapl = client.articles_for("AAPL")
        nvda = client.sentiment_for("NVDA")

        assert mock_get.call_count == 2
        assert [a["title"] for a in aapl] == ["Apple Q2 beats expectations"]
        assert nvda["news_volume_24h"] == 0
        assert client.degraded_reason == "av_fetch_error"
        # Log niesie tylko typ wyjątku — treść błędu sieci bywa z URL-em i kluczem.
        assert "ConnectionError" in caplog.text
        assert "Max retries" not in caplog.text
        assert "av-secret-key" not in caplog.text


class TestCryptoPrefix:
    """AV NEWS_SENTIMENT przyjmuje krypto tickery z prefiksem CRYPTO:
    (e.g. CRYPTO:BTC, CRYPTO:ETH). Klient mapuje wewnętrznie — caller
    używa czystych BTC/ETH, request idzie z prefiksem, parsowanie też.
    """

    def _ok_with_crypto_feed(self, btc_relevance: float = 0.9,
                             btc_sentiment: float = 0.3) -> MagicMock:
        response = MagicMock(spec=requests.Response)
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "feed": [
                {
                    "title": "BTC headline",
                    "url": "https://x.com/btc",
                    "source": "x.com",
                    "time_published": "20260601T100000",
                    "summary": "summary",
                    "ticker_sentiment": [
                        {
                            "ticker": "CRYPTO:BTC",
                            "relevance_score": str(btc_relevance),
                            "ticker_sentiment_score": str(btc_sentiment),
                        },
                    ],
                }
            ]
        }
        return response

    def test_request_sends_crypto_prefix(self, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=self._ok_with_crypto_feed(),
        )
        client = AlphaVantageClient(
            api_keys=["k"], symbols=["BTC"], crypto_symbols={"BTC"}
        )

        client.articles_for("BTC")

        # tickers w params powinno być CRYPTO:BTC, nie BTC
        call_params = mock_get.call_args.kwargs["params"]
        assert call_params["tickers"] == "CRYPTO:BTC"

    def test_articles_for_plain_ticker_returns_crypto_articles(self, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=self._ok_with_crypto_feed(),
        )
        client = AlphaVantageClient(
            api_keys=["k"], symbols=["BTC"], crypto_symbols={"BTC"}
        )

        articles = client.articles_for("BTC")

        assert len(articles) == 1
        assert articles[0]["title"] == "BTC headline"

    def test_sentiment_for_plain_crypto_ticker(self, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=self._ok_with_crypto_feed(
                btc_relevance=0.9, btc_sentiment=0.4
            ),
        )
        client = AlphaVantageClient(
            api_keys=["k"], symbols=["BTC"], crypto_symbols={"BTC"}
        )

        sentiment = client.sentiment_for("BTC")

        assert sentiment["news_volume_24h"] == 1
        assert sentiment["av_sentiment_score"] == pytest.approx(0.4)

    def test_equity_tickers_unaffected_by_crypto_set(self, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=self._ok_with_crypto_feed(),
        )
        client = AlphaVantageClient(
            api_keys=["k"],
            symbols=["BTC", "AAPL"],
            crypto_symbols={"BTC"},
        )

        client.articles_for("BTC")

        # AAPL pozostaje czysty, BTC dostał prefix.
        sent_tickers = set(
            mock_get.call_args_list[0].kwargs["params"]["tickers"].split(",")
        )
        assert "CRYPTO:BTC" in sent_tickers or sent_tickers == {"CRYPTO:BTC"}


class TestQuotaAlertEmit:
    """Wyczerpanie wszystkich kluczy AV → QuotaAlert(CRITICAL) do monitora."""

    def _rate_limited_response(self) -> MagicMock:
        response = MagicMock(spec=requests.Response)
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "Information": "Thank you for using Alpha Vantage! "
            "Our standard API rate limit is 25 requests per day."
        }
        return response

    def test_emits_critical_alert_when_all_keys_exhausted(self, mocker):
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        monitor = QuotaMonitor()
        client = AlphaVantageClient(
            api_keys=["k1", "k2"],
            symbols=["AAPL"],
            quota_monitor=monitor,
        )
        mocker.patch(
            "requests.Session.get",
            return_value=self._rate_limited_response(),
        )

        client.articles_for("AAPL")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert alert.source == "Alpha Vantage"
        assert alert.severity is QuotaSeverity.CRITICAL
        assert "2" in alert.message  # liczba kluczy
        assert "ALPHA_VANTAGE_API_KEYS" in alert.action

    def test_no_alert_when_keys_have_budget(self, mocker):
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        client = AlphaVantageClient(
            api_keys=["k1"],
            symbols=["AAPL"],
            quota_monitor=monitor,
        )
        response = MagicMock(spec=requests.Response)
        response.raise_for_status = MagicMock()
        response.json.return_value = {"feed": []}
        mocker.patch("requests.Session.get", return_value=response)

        client.articles_for("AAPL")

        assert monitor.alerts == []

    def test_no_monitor_means_no_emit_no_crash(self, mocker):
        # Bez monitora: rate-limit dalej działa, tylko brak emitu.
        client = AlphaVantageClient(api_keys=["k"], symbols=["AAPL"])
        mocker.patch(
            "requests.Session.get",
            return_value=self._rate_limited_response(),
        )

        client.articles_for("AAPL")  # nie wywala się


class TestRateLimitRotationDoesNotSleep:
    """Rotacja kluczy SPOWODOWANA rate-limitem nie powinna spać: do martwego
    klucza nie idzie żaden request, więc pacing 1 req/sec go nie dotyczy.
    Na wyczerpany dzień to oszczędza re-chodzenie po martwych kluczach."""

    def _rate_limit_response(self):
        return _ok_response({"Information": "rate limit reached"})

    def test_no_sleep_between_dead_keys_on_rotation(self, mocker):
        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mock_get = mocker.patch("requests.Session.get")
        # 3 klucze, wszystkie rate-limited dla jednego symbolu (1 batch).
        mock_get.side_effect = [self._rate_limit_response()] * 3

        client = AlphaVantageClient(
            api_keys=["k1", "k2", "k3"], symbols=["AAPL"]
        )
        client._fetch()

        # Rotacja po martwych kluczach: żaden request nie poszedł do nich
        # naprawdę → brak sleepów pacingowych.
        assert sleep_mock.call_count == 0

    def test_short_circuits_when_all_keys_exhausted_across_batches(self, mocker):
        # Po wyczerpaniu wszystkich kluczy w batchu 1 NIE re-chodzimy po nich
        # w batchu 2 — short-circuit, zero dodatkowych requestów i sleepów.
        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mock_get = mocker.patch("requests.Session.get")
        mock_get.side_effect = [self._rate_limit_response()] * 2

        symbols = [f"SYM{i:02d}" for i in range(15)]  # 2 batche @ batch_size=10
        client = AlphaVantageClient(
            api_keys=["k1", "k2"], symbols=symbols, batch_size=10
        )
        client._fetch()

        # Batch 1 wyczerpuje oba klucze (2 requesty). Batch 2 nie odpala
        # ŻADNEGO requestu — klucze już znane jako martwe.
        assert mock_get.call_count == 2
        # I żadnego sleepu (rotacja po dead-keyach + brak batcha 2).
        assert sleep_mock.call_count == 0

    def test_happy_path_pacing_preserved(self, mocker):
        # Realne requesty dalej rozstawione 1 req/sec: 2 batche → 1 sleep.
        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_PAYLOAD),
        )
        symbols = [f"SYM{i:02d}" for i in range(15)]
        client = AlphaVantageClient(
            api_keys=["k1", "k2"], symbols=symbols, batch_size=10
        )
        client._fetch()

        assert sleep_mock.call_count == 1
        assert sleep_mock.call_args.args[0] >= 1.0


class TestNewsWindow:
    """`news_volume_24h` ma liczyć artykuły z ostatnich 24 h, a nie „ile
    relevantnych wśród 50 najnowszych" (dla NVDA to godziny, dla IHI miesiące).

    Okno idzie do AV jako `time_from` (ta sama liczba requestów — FinOps 0),
    a filtr po `time_published` jest drugim bezpiecznikiem po stronie klienta."""

    _NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    @staticmethod
    def _article(url: str, published: str) -> dict:
        return {
            "title": f"AAPL news {url}",
            "url": url,
            "time_published": published,
            "summary": "...",
            "source": "Reuters",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.9",
                    "ticker_sentiment_score": "0.4",
                },
            ],
        }

    def _client(self) -> AlphaVantageClient:
        return AlphaVantageClient(
            api_keys=["av-test"],
            symbols=["AAPL"],
            news_window=timedelta(hours=24),
            clock=lambda: self._NOW,
        )

    def test_fetch_sends_time_from_param(self, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"feed": []}),
        )

        self._client().sentiment_for("AAPL")

        params = mock_get.call_args.kwargs["params"]
        # Format AV: YYYYMMDDTHHMM, okno = now − 24 h.
        assert params["time_from"] == "20261017T1200"
        assert params["sort"] == "LATEST"
        assert mock_get.call_count == 1

    def test_articles_older_than_window_are_dropped_from_volume(self, mocker):
        payload = {
            "feed": [
                self._article("https://example.com/fresh", "20261018T090000"),
                self._article("https://example.com/edge", "20261017T120000"),
                self._article("https://example.com/stale", "20261016T090000"),
            ]
        }
        mocker.patch("requests.Session.get", return_value=_ok_response(payload))
        client = self._client()

        sentiment = client.sentiment_for("AAPL")
        urls = [a["url"] for a in client.articles_for("AAPL")]

        assert sentiment["news_volume_24h"] == 2
        assert urls == ["https://example.com/fresh", "https://example.com/edge"]

    def test_article_without_parsable_timestamp_is_dropped(self, mocker):
        payload = {
            "feed": [
                self._article("https://example.com/fresh", "20261018T090000"),
                self._article("https://example.com/broken", "yesterday"),
            ]
        }
        mocker.patch("requests.Session.get", return_value=_ok_response(payload))

        sentiment = self._client().sentiment_for("AAPL")

        # Bez znacznika czasu nie da się wykazać, że artykuł jest z okna.
        assert sentiment["news_volume_24h"] == 1


def test_exhausted_keys_alert_is_polish_and_logged_in_english(mocker, caplog):
    # MAIL-15: komunikat alertu idzie do maila po polsku, diagnoza do logów po angielsku.
    import logging

    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    client = AlphaVantageClient(api_keys=["k1", "k2"], symbols=["AAPL"], quota_monitor=monitor)
    rate_limited = MagicMock(spec=requests.Response)
    rate_limited.raise_for_status = MagicMock()
    rate_limited.json.return_value = {
        "Information": "Thank you for using Alpha Vantage! "
        "Our standard API rate limit is 25 requests per day."
    }
    mocker.patch("requests.Session.get", return_value=rate_limited)

    with caplog.at_level(logging.WARNING):
        client.articles_for("AAPL")

    alert = monitor.alerts[0]
    assert "wyczerpały dzienny limit" in alert.message
    assert "incomplete" not in alert.message
    assert alert.action.startswith("Dodaj kolejny klucz")
    assert "Alpha Vantage API keys exhausted" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [{"Error Message": "Invalid API call. Please retry or visit the documentation."}, {}],
    ids=["error_message", "empty"],
)
def test_error_message_payload_sets_degraded_reason(mocker, caplog, payload) -> None:
    # DAT-07: payload bez `feed` i bez markera limitu to nie „spokojny dzień” —
    # sentyment niesie `degraded_reason`, ale TYLKO symbol z tym payloadem.
    import logging

    mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
    mocker.patch(
        "requests.Session.get",
        side_effect=[_ok_response(payload), _ok_response({"feed": []})],
    )
    client = AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL", "AMD"])

    with caplog.at_level(logging.ERROR):
        aapl = client.sentiment_for("AAPL")
    amd = client.sentiment_for("AMD")

    assert aapl["degraded_reason"] == "av_unexpected_payload"
    assert "degraded_reason" not in amd
    # Flaga cykliczna oflagowałaby wszystkie symbole.
    assert client.degraded_reason is None
    assert "AAPL" in caplog.text
    assert f"keys: {sorted(payload)}" in caplog.text


def test_unexpected_payload_reaches_the_sentiment_adapter_for_its_symbol_only(mocker) -> None:
    from src.infrastructure.adapters.alpha_vantage_adapters import (
        AlphaVantageSentimentAdapter,
    )

    mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
    mocker.patch(
        "requests.Session.get",
        side_effect=[
            _ok_response({"Error Message": "Invalid API call."}),
            _ok_response({"feed": []}),
        ],
    )
    adapter = AlphaVantageSentimentAdapter(
        AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL", "AMD"])
    )

    assert adapter.get_social_score("AAPL")["degraded_reason"] == "av_unexpected_payload"
    assert "degraded_reason" not in adapter.get_social_score("AMD")


def test_soft_limit_log_carries_the_marker_without_the_key(mocker, caplog) -> None:
    # DAT-06: diagnoza wymaga treści markera, a AV potrafi wkleić do niej klucz.
    import logging

    mocker.patch(
        "requests.Session.get",
        return_value=_ok_response({
            "Information": "We have detected your API key as news-secret-key and our "
            "standard API rate limit is 25 requests per day."
        }),
    )
    client = AlphaVantageClient(api_keys=["news-secret-key"], symbols=["AAPL"])

    with caplog.at_level(logging.WARNING):
        client.articles_for("AAPL")

    assert "standard API rate limit is 25 requests per day" in caplog.text
    assert "news-secret-key" not in caplog.text


def test_http_error_message_omits_apikey(mocker) -> None:
    # BEZ-07: wyjątek wypływa do grafu i dalej do maila / logów — bez `&apikey=`.
    import traceback

    response = requests.Response()
    response.status_code = 401
    response.reason = "Unauthorized"
    response.url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey=news-secret-key"
    mocker.patch("requests.Session.get", return_value=response)
    client = AlphaVantageClient(api_keys=["news-secret-key"], symbols=["AAPL"])

    with pytest.raises(requests.HTTPError) as excinfo:
        client.articles_for("AAPL")

    assert "401 Client Error" in str(excinfo.value)
    assert "news-secret-key" not in "".join(traceback.format_exception(excinfo.value))


def test_network_error_message_omits_apikey(mocker) -> None:
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            "Max retries exceeded with url: /query?function=NEWS_SENTIMENT&apikey=news-secret-key"
        ),
    )
    client = AlphaVantageClient(api_keys=["news-secret-key"], symbols=["AAPL"])

    with pytest.raises(requests.ConnectionError) as excinfo:
        client.articles_for("AAPL")

    assert "news-secret-key" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_supported_tickers_drops_symbols_alpha_vantage_rejects() -> None:
    # FIN-06: ta sama lista, z której klient pobiera feed, liczy requesty cyklu.
    from src.infrastructure.adapters.alpha_vantage_client import supported_tickers

    assert supported_tickers(["AAPL", "CSPX.L", "BTC", "XDWD.DE"]) == ["AAPL", "BTC"]


def _article(url: str, tickers: dict[str, str]) -> dict:
    """Artykuł z `ticker_sentiment` dla podanych tickerów (ticker → relevance)."""
    return {
        "title": f"News {url}",
        "url": url,
        "time_published": "20261018T090000",
        "summary": "...",
        "source": "Reuters",
        "ticker_sentiment": [
            {"ticker": ticker, "relevance_score": relevance, "ticker_sentiment_score": "0.3"}
            for ticker, relevance in tickers.items()
        ],
    }


class TestLazyPerSymbol:
    """FIN-04: tryb leniwy — request AV tylko dla symbolu, o który pyta graf
    (za bramką volatility), z cache per symbol, globalnym pacingiem i rotacją
    klucza odporną na współbieżność. Wsadowy pierwszy symbol pobierał feed dla
    WSZYSTKICH tickerów, więc „pod bramką nic płatnego” było fałszem per symbol."""

    _SYMBOLS = ["AAPL", "AMD", "NVDA"]

    def _client(self, **kwargs) -> AlphaVantageClient:
        options = {"api_keys": ["k1", "k2"], "symbols": self._SYMBOLS, **kwargs}
        return AlphaVantageClient(lazy_per_symbol=True, **options)

    def test_only_requested_symbol_is_fetched(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_ok_response(_SAMPLE_PAYLOAD)
        )
        client = self._client()

        articles = client.articles_for("AMD")
        sentiment = client.sentiment_for("AMD")

        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["tickers"] == "AMD"
        assert [a["title"] for a in articles] == ["AMD surges on AI chip demand"]
        assert sentiment["news_volume_24h"] == 1

    def test_feed_is_cached_per_symbol(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_ok_response({"feed": []})
        )
        client = self._client()

        client.sentiment_for("AAPL")
        client.articles_for("AAPL")
        client.sentiment_for("AMD")
        client.articles_for("AAPL")

        tickers_sent = [c.kwargs["params"]["tickers"] for c in mock_get.call_args_list]
        assert tickers_sent == ["AAPL", "AMD"]

    def test_volume_counts_only_the_symbols_own_feed(self, mocker):
        # Rozkład `news_volume_24h` się zmienia (FEATURE_SET_VERSION): artykuł
        # o AAPL z feedu AMD nie dolicza się już do AAPL.
        feeds = {
            "AAPL": {"feed": [_article("https://example.com/a1", {"AAPL": "0.9"})]},
            "AMD": {
                "feed": [_article("https://example.com/m1", {"AMD": "0.9", "AAPL": "0.9"})]
            },
        }
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get",
            side_effect=lambda *_a, **kw: _ok_response(feeds[kw["params"]["tickers"]]),
        )
        client = self._client()

        client.sentiment_for("AMD")

        assert client.sentiment_for("AAPL")["news_volume_24h"] == 1

    def test_unconfigured_or_rejected_ticker_makes_no_request(self, mocker):
        mock_get = mocker.patch("requests.Session.get")
        client = self._client(symbols=["AAPL", "CSPX.L"])

        assert client.articles_for("CSPX.L") == []
        assert client.sentiment_for("TSLA")["news_volume_24h"] == 0
        assert mock_get.call_count == 0

    def test_requests_are_paced_one_per_interval(self, mocker):
        from src.infrastructure.adapters.alpha_vantage_client import (
            INTER_REQUEST_DELAY_SECONDS,
        )

        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mocker.patch("requests.Session.get", return_value=_ok_response({"feed": []}))
        client = self._client()

        for symbol in self._SYMBOLS:
            client.sentiment_for(symbol)

        # Pierwszy request bez czekania, każdy kolejny po ~1 interwale.
        waits = [c.args[0] for c in sleep_mock.call_args_list]
        assert waits == [pytest.approx(INTER_REQUEST_DELAY_SECONDS, abs=0.05)] * 2

    def test_parallel_symbols_share_one_pacer(self, mocker):
        import threading

        sleep_mock = mocker.patch(
            "src.infrastructure.adapters.alpha_vantage_client.time.sleep"
        )
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_ok_response({"feed": []})
        )
        client = self._client()
        barrier = threading.Barrier(len(self._SYMBOLS))

        def worker(symbol: str) -> None:
            barrier.wait()
            client.sentiment_for(symbol)

        threads = [threading.Thread(target=worker, args=(s,)) for s in self._SYMBOLS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert mock_get.call_count == 3
        assert sleep_mock.call_count == 2

    def test_concurrent_callers_of_one_symbol_fetch_it_once(self, mocker):
        import threading
        import time as real_time

        def slow_get(*_args, **_kwargs):
            real_time.sleep(0.02)  # poszerza okno race'a
            return _ok_response({"feed": []})

        mock_get = mocker.patch("requests.Session.get", side_effect=slow_get)
        client = self._client()
        barrier = threading.Barrier(6)

        def worker() -> None:
            barrier.wait()
            client.sentiment_for("AAPL")

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert mock_get.call_count == 1

    def test_concurrent_rate_limits_rotate_the_key_once(self, mocker):
        # Dwa symbole równolegle dostają limit na key1. Bez locka rotacji oba
        # przesuwają indeks i przeskakują żywy key2 — cykl kończy się „exhausted”.
        import threading

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        both_in_flight = threading.Barrier(2, timeout=5)

        def get(*_args, **kwargs):
            if kwargs["params"]["apikey"] == "k1":
                both_in_flight.wait()
                return _ok_response({"Information": "rate limit reached"})
            return _ok_response({"feed": []})

        mock_get = mocker.patch("requests.Session.get", side_effect=get)
        client = self._client()
        results: dict[str, dict] = {}

        def worker(symbol: str) -> None:
            results[symbol] = client.sentiment_for(symbol)

        threads = [threading.Thread(target=worker, args=(s,)) for s in ("AAPL", "AMD")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        keys_used = sorted(c.kwargs["params"]["apikey"] for c in mock_get.call_args_list)
        assert keys_used == ["k1", "k1", "k2", "k2"]
        assert client._active_key_idx == 1
        assert all("degraded_reason" not in r for r in results.values())

    def test_exhausted_keys_return_empty_without_http(self, mocker):
        from src.application.quota_monitor import QuotaMonitor

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"Information": "rate limit reached"}),
        )
        monitor = QuotaMonitor()
        client = self._client(quota_monitor=monitor)

        aapl = client.sentiment_for("AAPL")   # key1 + key2 → wyczerpane
        amd = client.sentiment_for("AMD")     # zero HTTP

        assert mock_get.call_count == 2
        assert aapl["degraded_reason"] == amd["degraded_reason"] == "av_keys_exhausted"
        assert len(monitor.alerts) == 1

    def test_symbol_fetched_before_exhaustion_keeps_clean_data(self, mocker):
        # Flaga cykliczna oflagowałaby też AAPL, który ma pełny feed.
        from src.infrastructure.adapters.alpha_vantage_adapters import (
            AlphaVantageSentimentAdapter,
        )

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response({"feed": []}),                          # AAPL / k1
                _ok_response({"Information": "rate limit reached"}),  # AMD / k1
                _ok_response({"Information": "rate limit reached"}),  # AMD / k2
            ],
        )
        client = self._client()
        adapter = AlphaVantageSentimentAdapter(client)

        aapl = adapter.get_social_score("AAPL")
        amd = adapter.get_social_score("AMD")

        assert "degraded_reason" not in adapter.get_social_score("AAPL")
        assert "degraded_reason" not in aapl
        assert amd["degraded_reason"] == "av_keys_exhausted"
        assert client.degraded_reason is None

    def test_fetch_error_flags_its_symbol_and_is_not_refetched(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[requests.ConnectionError("down"), _ok_response({"feed": []})],
        )
        client = self._client()

        with pytest.raises(requests.ConnectionError):
            client.sentiment_for("AAPL")
        aapl = client.sentiment_for("AAPL")
        amd = client.sentiment_for("AMD")

        assert mock_get.call_count == 2
        assert aapl["degraded_reason"] == "av_fetch_error"
        assert "degraded_reason" not in amd

    def test_fetch_error_does_not_mute_a_later_exhausted_keys_alert(self, mocker):
        # DAT-08: wcześniejszy `av_fetch_error` nie może wyciszyć CRITICAL o kluczach.
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                requests.ConnectionError("down"),
                _ok_response({"Information": "rate limit reached"}),
                _ok_response({"Information": "rate limit reached"}),
            ],
        )
        monitor = QuotaMonitor()
        client = self._client(quota_monitor=monitor)

        with pytest.raises(requests.ConnectionError):
            client.sentiment_for("AAPL")
        client.sentiment_for("AMD")

        assert [a.severity for a in monitor.alerts] == [QuotaSeverity.CRITICAL]

    def test_unexpected_payload_flags_its_symbol(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response({"Error Message": "Invalid API call."}),
                         _ok_response({"feed": []})],
        )
        client = self._client()

        assert client.sentiment_for("AAPL")["degraded_reason"] == "av_unexpected_payload"
        assert "degraded_reason" not in client.sentiment_for("AMD")

    def test_all_symbols_of_a_cycle_share_one_news_window(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_ok_response({"feed": []})
        )
        ticks = iter([
            datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
            datetime(2026, 10, 18, 12, 9, tzinfo=UTC),
        ])
        client = self._client(news_window=timedelta(hours=24), clock=lambda: next(ticks))

        client.sentiment_for("AAPL")
        client.sentiment_for("AMD")

        windows = {c.kwargs["params"]["time_from"] for c in mock_get.call_args_list}
        assert windows == {"20261017T1200"}

    def test_crypto_symbol_is_fetched_with_its_prefix(self, mocker):
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_ok_response({"feed": []})
        )
        client = self._client(symbols=["AAPL", "BTC"], crypto_symbols={"BTC"})

        client.sentiment_for("BTC")

        assert mock_get.call_args.kwargs["params"]["tickers"] == "CRYPTO:BTC"
