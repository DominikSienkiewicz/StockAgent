"""MAIL-32: dostawca, którego adapter zgłosił błąd symbolu — do maila obok kategorii.

Taksonomia portów (ARCH-33) mówi, CO się stało, ale nie U KOGO: `ProviderUnavailable`
z ceny i z newsów to ta sama klasa. Dostawcę wskazuje najgłębsza ramka tracebacku
w module adaptera ze znanym dostawcą — adapter rzucił sam (`MalformedPayload`,
`NotFound`) albo przetłumaczył wyjątek biblioteki; w obu przypadkach jego ramka jest
na stosie. Wspólne moduły (`_http.py`, wrappery i routery LLM, `write_guard`) nie
mają dostawcy, więc nie przesłaniają adaptera. Wyjątek opakowany wyżej
(`raise … from exc`) oddaje dostawcę swojej przyczyny.

Nazwy dostawców jak w `_http._HOST_SOURCES` (banner kwot, stan źródeł MAIL-21).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

_ADAPTERS = "src.infrastructure.adapters"
_LLM = "src.infrastructure.llm"

PROVIDER_BY_MODULE: Mapping[str, str] = {
    f"{_ADAPTERS}._finnhub_common": "Finnhub",
    f"{_ADAPTERS}.finnhub_api": "Finnhub",
    f"{_ADAPTERS}.finnhub_analyst": "Finnhub",
    f"{_ADAPTERS}.finnhub_earnings": "Finnhub",
    f"{_ADAPTERS}.finnhub_options": "Finnhub",
    f"{_ADAPTERS}._alpha_vantage_common": "Alpha Vantage",
    f"{_ADAPTERS}.alpha_vantage_adapters": "Alpha Vantage",
    f"{_ADAPTERS}.alpha_vantage_client": "Alpha Vantage",
    f"{_ADAPTERS}.alpha_vantage_earnings": "Alpha Vantage",
    f"{_ADAPTERS}.alpha_vantage_fundamentals": "Alpha Vantage",
    f"{_ADAPTERS}.coingecko": "CoinGecko",
    f"{_ADAPTERS}.edgar_insider": "EDGAR",
    f"{_ADAPTERS}.reddit_social": "Reddit",
    f"{_ADAPTERS}.fred_client": "FRED",
    f"{_ADAPTERS}.nbp_client": "NBP",
    f"{_ADAPTERS}.deribit_dvol": "Deribit",
    f"{_ADAPTERS}.supabase_repo": "Supabase",
    f"{_ADAPTERS}.positions_repo": "Supabase",
    f"{_ADAPTERS}.supabase_storage_archive": "Supabase",
    f"{_LLM}.anthropic_client": "Anthropic",
    f"{_LLM}.tool_calling_anthropic": "Anthropic",
    f"{_LLM}.openai_client": "OpenAI",
    f"{_LLM}.tool_calling_openai": "OpenAI",
    f"{_LLM}.openai_embeddings": "OpenAI",
}


def provider_of(exc: BaseException) -> str | None:
    """Dostawca z najgłębszej ramki adaptera na stosie `exc`, a gdy jej nie ma —
    z łańcucha przyczyn (`__cause__`, `__context__`). None = błąd poza adapterami."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        provider = _deepest_provider(current.__traceback__)
        if provider is not None:
            return provider
        current = current.__cause__ or current.__context__
    return None


def _deepest_provider(frames: TracebackType | None) -> str | None:
    provider: str | None = None
    while frames is not None:
        module = frames.tb_frame.f_globals.get("__name__")
        if isinstance(module, str) and module in PROVIDER_BY_MODULE:
            provider = PROVIDER_BY_MODULE[module]
        frames = frames.tb_next
    return provider
