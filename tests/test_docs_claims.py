"""Strażnik twierdzeń w docs i komentarzach, które kod już obalił (DOC-20).

Od migracji 012 trening XGBoost uczy się na `target_return` z widoku
`ml_feature_store`, a `accuracy_score` (bliskość ceny do celu) liczy się dalej,
ale trenować już nie służy. Komentarze w `ports.py` i `agent_graph.py` oraz
`docs/how-it-works.md` długo twierdziły coś przeciwnego — recenzent, który
przechodzi z opisu do kodu treningu, wyłapuje taki rozjazd w minutę.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

CLAIM_SOURCES = ("README.md", "docs/*.md", "src/**/*.py", "main_*.py", "config.toml")

# `accuracy_score`, a w ciągu kolejnych ~160 znaków (także przez koniec linii
# komentarza) „napędza trening” albo „feeds XGBoost training”. Okno zamiast jednej
# linii, bo komentarz w agent_graph.py zawijał to twierdzenie na dwie linie.
ACCURACY_FEEDS_TRAINING = re.compile(
    r"accuracy_score[\s\S]{0,160}?"
    r"(?:(?:napędza|zasila|karmi)\s+trening|feeds?\s+(?:the\s+)?(?:XGBoost\s+)?training)",
    re.IGNORECASE,
)


def _claim_files() -> list[Path]:
    return sorted(
        path for pattern in CLAIM_SOURCES for path in ROOT.glob(pattern) if path.is_file()
    )


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def test_docs_do_not_claim_accuracy_score_feeds_training() -> None:
    claims = [
        f"{path.relative_to(ROOT)}:{_line_of(text, found.start())}"
        for path in _claim_files()
        for text in [path.read_text(encoding="utf-8")]
        for found in ACCURACY_FEEDS_TRAINING.finditer(text)
    ]

    assert claims == [], (
        "accuracy_score nie napędza treningu — uczy się na `target_return` z "
        f"ml_feature_store: {claims}"
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "`accuracy_score` (how close the price landed — feeds XGBoost training)",
        "# Domena liczy accuracy_score (0.0-1.0) — cena do celu,\n# to napędza trening XGBoost.",
        "`accuracy_score` (bliskość ceny do\n celu) napędza trening;",
    ],
)
def test_accuracy_claim_pattern_matches_both_languages_and_wrapped_lines(sentence: str) -> None:
    assert ACCURACY_FEEDS_TRAINING.search(sentence)


def test_accuracy_claim_pattern_ignores_the_corrected_wording() -> None:
    corrected = (
        "`accuracy_score` feeds the weekly recap; training uses `target_return` "
        "from `ml_feature_store`"
    )

    assert not ACCURACY_FEEDS_TRAINING.search(corrected)


# DOC-35: dwa zdania, które każdy da się sprawdzić jednym grepem, a mimo to
# kłamały — RAG działa od migracji 011, a „default portfolio of 43” to lista z
# config.toml, nie domyślna wartość kodu (ta ma trzy tickery).
def test_ports_do_not_call_the_rag_similarity_search_a_future_feature() -> None:
    text = (ROOT / "src" / "application" / "ports.py").read_text(encoding="utf-8")

    assert not re.search(r"przyszł\w*\s+similarity", text), (
        "find_similar_predictions działa od migracji 011 (RAG w predict_node)"
    )


def _how_it_works() -> str:
    return (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")


def test_how_it_works_symbols_section_lists_the_shipped_watchlist() -> None:
    import tomllib

    text = _how_it_works()
    start = text.index("\n## Symbols")
    section = text[start : text.index("\n## ", start + 1)]
    heading = section.strip().splitlines()[0]
    block = re.search(r"```text\n(.*?)```", section, re.S)
    assert block is not None, "sekcja Symbols nie ma bloku z listą"
    listed = [symbol.strip() for symbol in block.group(1).replace("\n", "").split(",") if symbol]
    shipped = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["symbols"]

    assert "default" not in heading.lower(), (
        f"{heading!r}: lista to watchlista z config.toml, domyślne trzy tickery to smoke-test"
    )
    assert listed == shipped


def test_how_it_works_names_the_code_default_of_symbols() -> None:
    from src.config import Settings

    factory = Settings.model_fields["symbols"].default_factory
    assert factory is not None
    smoke_portfolio = ", ".join(factory())  # type: ignore[call-arg]

    assert f"`{smoke_portfolio}`" in _how_it_works()


# DOC-35: dwie linie bloku komend w running-locally.md były sklejone z kilku
# poleceń i z duplikatami (scalenie dwóch gałęzi dokleiło sąsiednie wiersze
# zamiast je rozdzielić). Blok komend ma mieć jedno polecenie na linię i żadnego
# polecenia dwa razy.
def _fenced_blocks(path: Path) -> list[list[tuple[int, str]]]:
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append((number, line))
    return blocks


def _command_of(line: str) -> str:
    return line.split("#", 1)[0].strip()


@pytest.mark.parametrize("pattern", ["README.md", "docs/*.md"])
def test_command_blocks_hold_one_uv_command_per_line(pattern: str) -> None:
    glued = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(ROOT.glob(pattern))
        for block in _fenced_blocks(path)
        for number, line in block
        if line.count("uv run ") > 1
    ]

    assert glued == [], f"kilka poleceń w jednej linii bloku komend: {glued}"


@pytest.mark.parametrize("pattern", ["README.md", "docs/*.md"])
def test_command_blocks_do_not_repeat_a_command(pattern: str) -> None:
    repeated = []
    for path in sorted(ROOT.glob(pattern)):
        for block in _fenced_blocks(path):
            seen: dict[str, int] = {}
            for number, line in block:
                command = _command_of(line)
                if command.startswith("uv run") and command in seen:
                    where = f"{path.relative_to(ROOT)}:{number}"
                    repeated.append(f"{where} (jak w linii {seen[command]})")
                seen.setdefault(command, number)

    assert repeated == [], f"to samo polecenie dwa razy w jednym bloku: {repeated}"


# DOC-34: tabela „Sources” w how-it-works.md nie wymieniała części źródeł, które
# działają w produkcji (EDGAR, Reddit, FRED, analitycy Finnhub), ani tych za
# flagami off (option-chain, kalendarz Finnhub, Deribit). Każda flaga źródła
# danych ma w tabeli wiersz z nazwą dostawcy i nazwą tej flagi, żeby czytelnik
# widział, czym ją włączyć i co ją kosztuje.
DATA_SOURCE_FLAGS = {
    "insider_flow_enabled": ("SEC EDGAR",),
    "social_velocity_enabled": ("Reddit",),
    "yield_curve_enabled": ("FRED",),
    "macro_series_enabled": ("FRED",),
    "analyst_consensus_enabled": ("Finnhub", "/stock/recommendation"),
    "options_flow_enabled": ("Finnhub", "option-chain"),
    "finnhub_earnings_calendar_enabled": ("Finnhub", "/calendar/earnings"),
    "earnings_calendar_enabled": ("Alpha Vantage", "EARNINGS_CALENDAR"),
    "deribit_dvol_enabled": ("Deribit",),
    "nbp_enabled": ("NBP",),
}


def _sources_rows() -> list[str]:
    text = _how_it_works()
    start = text.index("\n## Sources")
    section = text[start : text.index("\n## ", start + 1)]
    return [line for line in section.splitlines() if line.startswith("| **")]


def test_data_source_flags_are_settings_fields() -> None:
    from src.config import Settings

    missing = [flag for flag in DATA_SOURCE_FLAGS if flag not in Settings.model_fields]

    assert missing == [], f"flaga źródła zniknęła z Settings — zaktualizuj mapę: {missing}"


@pytest.mark.parametrize("flag, names", sorted(DATA_SOURCE_FLAGS.items()))
def test_sources_table_has_a_row_for_every_data_source_flag(
    flag: str, names: tuple[str, ...]
) -> None:
    rows = [row for row in _sources_rows() if f"`{flag}`" in row]

    assert rows, f"tabela Sources nie wymienia flagi `{flag}`"
    assert any(all(name in row for name in names) for row in rows), (
        f"wiersz z `{flag}` nie nazywa dostawcy {names}"
    )


def test_sources_table_has_no_duplicate_rows() -> None:
    keys = [tuple(cell.strip() for cell in row.split("|")[1:3]) for row in _sources_rows()]
    duplicated = sorted({key for key in keys if keys.count(key) > 1})

    assert duplicated == [], f"ten sam wiersz tabeli Sources dwa razy: {duplicated}"


def test_sources_rows_saying_off_by_default_name_flags_that_default_to_false() -> None:
    from src.config import Settings

    wrong: list[str] = []
    for row in _sources_rows():
        if "Off by default" not in row:
            continue
        flags = re.findall(r"`(\w+_enabled)`", row)
        wrong.extend(
            flag
            for flag in flags
            if flag not in Settings.model_fields or Settings.model_fields[flag].default is not False
        )

    assert wrong == [], f"„Off by default” przy fladze, która w kodzie jest włączona: {wrong}"


def test_sources_rows_stating_a_config_toml_state_match_config_toml() -> None:
    import tomllib

    shipped = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    checked = 0
    wrong = []
    for row in _sources_rows():
        for stated in re.finditer(r"\b(On|Off) in `config\.toml`", row):
            # Stan dotyczy ostatniej flagi wymienionej przed zdaniem o nim.
            flag = re.findall(r"`(\w+_enabled)`", row[: stated.start()])[-1]
            checked += 1
            says_on = stated.group(1) == "On"
            if shipped.get(flag) is not says_on:
                wrong.append(
                    f"{flag}: tabela mówi {stated.group(1)}, config.toml {shipped.get(flag)!r}"
                )

    assert checked >= len(DATA_SOURCE_FLAGS) - 2, "wzorzec stanu przestał łapać wiersze"
    assert wrong == []


# DOC-41: polityka degradacji żyła tylko w komentarzach. Tabela „Failure modes” w
# how-it-works ma wiersz dla każdej flagi jakości, którą kod zostawia na rekordzie, i
# uczciwie pokazuje „raport niedostarczony → exit 0”; nowa flaga bez wiersza czerwieni
# test. Flagi zbieramy ze stałych `*_FLAG` w `src/`, z rejestrów flag (trening, stan
# źródeł) i z literałów oraz szablonów pól w `agent_graph.py`.
FAILURE_MODES_HEADING = "## Failure modes"
AGENT_GRAPH = ROOT / "src" / "application" / "agent_graph.py"
_FLAG_CONSTANT = re.compile(
    r'^_?[A-Z][A-Z0-9_]*_FLAG(?:\s*:\s*\w+)?\s*=\s*"([a-z][a-z0-9_]*)"', re.MULTILINE
)
_APPENDED_FLAG = re.compile(r'flags\.append\("([a-z][a-z0-9_]*)"\)')
_FIELD_FLAG_TEMPLATE = re.compile(r'f"\{\w+\}_([a-z][a-z0-9_]*)"')
# Wiersze, których katalog (DOC-41) wymaga z nazwy, nie tylko przez wzorzec `*_…`.
NAMED_FAILURE_ROWS = (
    "llm_analysis_failed",
    "av_keys_exhausted",
    "price_delta_no_prior_prediction",
    "trend_direction_invalid",
    "*_missing",
)


def _quality_flags_in_code() -> tuple[set[str], set[str]]:
    """(nazwy flag, wzorce `*_sufiks` z szablonów pól) — to, co kod zapisuje na rekordzie."""
    from src.application.ml_features import TRAINING_EXCLUDED_QUALITY_FLAGS
    from src.application.source_health import LLM_DEGRADATION_FLAGS

    graph = AGENT_GRAPH.read_text(encoding="utf-8")
    named = {
        flag
        for path in sorted((ROOT / "src").rglob("*.py"))
        for flag in _FLAG_CONSTANT.findall(path.read_text(encoding="utf-8"))
    }
    named |= set(TRAINING_EXCLUDED_QUALITY_FLAGS) | set(LLM_DEGRADATION_FLAGS)
    named |= set(_APPENDED_FLAG.findall(graph))
    patterns = {f"*_{suffix}" for suffix in _FIELD_FLAG_TEMPLATE.findall(graph)}
    return named, patterns


def _failure_rows() -> list[list[str]]:
    text = _how_it_works()
    start = text.find(FAILURE_MODES_HEADING)
    if start == -1:
        return []
    end = text.find("\n## ", start + 1)
    section = text[start:] if end == -1 else text[start:end]
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| What failed")
    ]


def _documented_flags() -> set[str]:
    return {
        token for row in _failure_rows() for token in re.findall(r"`([a-z*][a-z0-9_*]*)`", row[1])
    }


def _covered(flag: str, documented: set[str]) -> bool:
    return flag in documented or any(
        token.startswith("*_") and flag.endswith(token[1:]) for token in documented
    )


def test_quality_flag_sources_find_the_known_flags() -> None:
    named, patterns = _quality_flags_in_code()

    assert {"llm_analysis_failed", "av_keys_exhausted", "price_delta_no_prior_prediction"} <= named
    assert {"*_missing", "*_invalid", "*_out_of_range"} <= patterns


def test_failure_modes_table_lists_every_quality_flag() -> None:
    named, patterns = _quality_flags_in_code()
    documented = _documented_flags()

    missing = sorted(
        {flag for flag in named if not _covered(flag, documented)}
        | {pattern for pattern in patterns if pattern not in documented}
        | {row for row in NAMED_FAILURE_ROWS if row not in documented}
    )
    assert missing == [], f"tabela Failure modes w how-it-works.md nie ma wierszy dla: {missing}"


def test_failure_modes_table_links_the_exit_code_table_and_shows_delivery_exit_zero() -> None:
    text = _how_it_works()
    start = text.find(FAILURE_MODES_HEADING)
    section = "" if start == -1 else text[start:].split("\n## ", 1)[0]
    delivery = [row for row in _failure_rows() if "Resend" in row[0]]

    assert "](running-locally.md#github-actions)" in section, "brak linku do tabeli exit code"
    assert delivery, "brak wiersza o niedostarczonym raporcie (Resend)"
    assert all(row[3].startswith("0") for row in delivery), delivery
