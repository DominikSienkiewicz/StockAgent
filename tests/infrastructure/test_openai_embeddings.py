from unittest.mock import MagicMock

import pytest

from src.application.ports import EmbeddingPort
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter


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
