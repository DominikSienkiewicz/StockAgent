"""Testy `AlphaVantageClient` — wspólny klient dla News i Sentiment adapterów."""

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
        assert article["relevance_score"] == 0.95
        assert article["ticker_sentiment_score"] == 0.52


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

        assert result["av_sentiment_score"] == 0.0
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
            t for t in mock_get.call_args_list[0].kwargs["params"]["tickers"].split(",")
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
