from unittest.mock import MagicMock

import pytest

from src.application.ports import EmbeddingPort
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter


def _rate_limit_error(retry_after: str | None = None):
    """Buduje openai.RateLimitError z realnym httpx.Response."""
    import httpx
    from openai import RateLimitError

    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def mock_openai_class(mocker):
    return mocker.patch("src.infrastructure.llm.openai_embeddings.OpenAI")


@pytest.fixture
def adapter_with_client(mock_openai_class) -> tuple[OpenAIEmbeddingAdapter, MagicMock]:
    client = MagicMock()
    mock_openai_class.return_value = client
    adapter = OpenAIEmbeddingAdapter(api_key="sk-test")
    return adapter, client


def _embedding_response(vector: list[float]) -> MagicMock:
    response = MagicMock()
    item = MagicMock()
    item.embedding = vector
    response.data = [item]
    return response


class TestEmbed:
    def test_implements_port(self):
        assert issubclass(OpenAIEmbeddingAdapter, EmbeddingPort)

    def test_returns_vector_from_api(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.embeddings.create.return_value = _embedding_response([0.1, 0.2, 0.3])

        result = adapter.embed("Apple beats earnings")

        assert result == [0.1, 0.2, 0.3]

    def test_uses_text_embedding_3_small_by_default(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.embeddings.create.return_value = _embedding_response([0.0])

        adapter.embed("some text")

        assert client.embeddings.create.call_args.kwargs["model"] == "text-embedding-3-small"

    def test_passes_text_as_input(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.embeddings.create.return_value = _embedding_response([0.0])

        adapter.embed("Nvidia surges on AI demand")

        assert client.embeddings.create.call_args.kwargs["input"] == "Nvidia surges on AI demand"

    def test_empty_text_returns_empty_vector_without_api_call(self, adapter_with_client):
        adapter, client = adapter_with_client

        assert adapter.embed("") == []
        assert adapter.embed("   ") == []
        client.embeddings.create.assert_not_called()


class TestConfiguration:
    def test_initializes_client_with_timeout(self, mock_openai_class):
        # GHA fast loop ma 15 min hard timeout — SDK domyślnie czeka 600s.
        # Embeddingi to płatny call na hot pathcie, więc muszą mieć krótszy timeout.
        OpenAIEmbeddingAdapter(api_key="sk-test")

        call = mock_openai_class.call_args
        assert call.kwargs["timeout"] > 0

    def test_custom_timeout_forwarded_to_client(self, mock_openai_class):
        OpenAIEmbeddingAdapter(api_key="sk-test", timeout=12.0)

        assert mock_openai_class.call_args.kwargs["timeout"] == pytest.approx(12.0)


class TestRetryAndQuota:
    """Embeddingi to płatny call na hot pathcie — muszą mieć tę samą odporność
    co OpenAIAdapter: retry na 429 z backoffem + sygnalizacja do QuotaMonitor."""

    def test_retries_on_rate_limit_then_succeeds(self, mock_openai_class, mocker):
        mocker.patch("src.infrastructure.llm.openai_embeddings.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.embeddings.create.side_effect = [
            _rate_limit_error(),
            _embedding_response([0.1, 0.2, 0.3]),
        ]
        adapter = OpenAIEmbeddingAdapter(api_key="sk-test", backoff_base=0.01)

        result = adapter.embed("Apple beats earnings")

        assert result == [0.1, 0.2, 0.3]
        assert client.embeddings.create.call_count == 2

    def test_warning_alert_emitted_after_successful_retry(
        self, mock_openai_class, mocker
    ):
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        mocker.patch("src.infrastructure.llm.openai_embeddings.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.embeddings.create.side_effect = [
            _rate_limit_error(),
            _embedding_response([0.0]),
        ]
        monitor = QuotaMonitor()
        adapter = OpenAIEmbeddingAdapter(
            api_key="sk-test", backoff_base=0.01, quota_monitor=monitor
        )

        adapter.embed("some text")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert alert.severity is QuotaSeverity.WARNING
        assert "embeddings" in alert.source.lower()

    def test_critical_alert_and_raise_when_retries_exhausted(
        self, mock_openai_class, mocker
    ):
        from openai import RateLimitError

        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        mocker.patch("src.infrastructure.llm.openai_embeddings.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.embeddings.create.side_effect = _rate_limit_error()
        monitor = QuotaMonitor()
        adapter = OpenAIEmbeddingAdapter(
            api_key="sk-test",
            backoff_base=0.01,
            max_retries=2,
            quota_monitor=monitor,
        )

        with pytest.raises(RateLimitError):
            adapter.embed("some text")

        assert client.embeddings.create.call_count == 3  # 1 + 2 retries
        assert monitor.max_severity() is QuotaSeverity.CRITICAL

    def test_honors_retry_after_header(self, mock_openai_class, mocker):
        sleep = mocker.patch("src.infrastructure.llm.openai_embeddings.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.embeddings.create.side_effect = [
            _rate_limit_error(retry_after="5"),
            _embedding_response([0.0]),
        ]
        adapter = OpenAIEmbeddingAdapter(api_key="sk-test", backoff_base=0.01)

        adapter.embed("some text")

        sleep.assert_called_once_with(5.0)


class TestContentHashCache:
    """#29 — Cross-cycle cache: ten sam wolnozmienny news_summary (cadence 12h)
    nie może być re-embedowany od zera w każdym cyklu. Cache po sha256
    znormalizowanego tekstu eliminuje czysty re-burn płatnego calla."""

    def test_same_text_twice_calls_api_once(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.embeddings.create.return_value = _embedding_response([0.1, 0.2])

        first = adapter.embed("Apple beats earnings")
        second = adapter.embed("Apple beats earnings")

        assert first == [0.1, 0.2]
        assert second == [0.1, 0.2]
        # Drugie wywołanie to cache hit — API ruszyło dokładnie raz.
        assert client.embeddings.create.call_count == 1

    def test_different_text_calls_api_twice(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.embeddings.create.side_effect = [
            _embedding_response([0.1]),
            _embedding_response([0.2]),
        ]

        adapter.embed("Apple beats earnings")
        adapter.embed("Nvidia surges on AI demand")

        assert client.embeddings.create.call_count == 2

    def test_empty_text_returns_empty_without_call_and_no_caching(
        self, adapter_with_client
    ):
        adapter, client = adapter_with_client

        assert adapter.embed("") == []
        assert adapter.embed("   ") == []
        client.embeddings.create.assert_not_called()

    def test_cache_is_bounded_oldest_evicted_past_capacity(self, mock_openai_class):
        client = MagicMock()
        mock_openai_class.return_value = client
        client.embeddings.create.side_effect = [
            _embedding_response([float(i)]) for i in range(10)
        ]
        # Pojemność 2 — po dodaniu trzeciego tekstu najstarszy wypada.
        adapter = OpenAIEmbeddingAdapter(api_key="sk-test", cache_size=2)

        adapter.embed("text-A")  # call 1
        adapter.embed("text-B")  # call 2
        adapter.embed("text-C")  # call 3 — wypycha text-A z cache'a
        assert client.embeddings.create.call_count == 3

        # text-B i text-C nadal w cache'u → bez nowych calli.
        adapter.embed("text-B")
        adapter.embed("text-C")
        assert client.embeddings.create.call_count == 3

        # text-A został eksmitowany → ponowny embedding to nowy call.
        adapter.embed("text-A")
        assert client.embeddings.create.call_count == 4

    def test_normalized_whitespace_hits_same_cache_entry(self, adapter_with_client):
        # Ten sam tekst z innym otaczającym whitespace'em → jeden wpis cache'a.
        adapter, client = adapter_with_client
        client.embeddings.create.return_value = _embedding_response([0.5])

        adapter.embed("Apple beats earnings")
        adapter.embed("  Apple beats earnings  ")

        assert client.embeddings.create.call_count == 1


@pytest.mark.integration
class TestOpenAIEmbeddingsLive:
    def test_real_api_returns_1536_dim_vector(self):
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        adapter = OpenAIEmbeddingAdapter(api_key=api_key)
        vector = adapter.embed("Apple Q2 earnings beat expectations")
        # text-embedding-3-small → 1536 wymiarów (pasuje do VECTOR(1536) w schemacie)
        assert len(vector) == 1536
