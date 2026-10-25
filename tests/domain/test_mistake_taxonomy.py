"""WIA-57 — taksonomia błędów diagnozy zamiast „najczęstszego słowa”.

Diagnoza LLM zwraca JSON `{category, insight}`; port `analyze_mistake` dalej
oddaje tekst, więc parse ma fallback: wszystko, co nie jest poprawnym JSON-em
z kategorią ze zbioru, to OTHER z całym tekstem jako diagnozą."""

from __future__ import annotations

import json

import pytest

from src.domain.mistake_taxonomy import (
    MISTAKE_CATEGORY_LABELS_PL,
    MistakeCategory,
    MistakeDiagnosis,
    count_categories,
    parse_batch_diagnoses,
    parse_mistake_category,
    parse_mistake_diagnosis,
)


class TestParseMistakeDiagnosis:
    def test_json_answer_gives_category_and_insight(self) -> None:
        raw = '{"category": "MACRO", "insight": "Zignorowałem decyzję Fed."}'

        assert parse_mistake_diagnosis(raw) == MistakeDiagnosis(
            MistakeCategory.MACRO, "Zignorowałem decyzję Fed."
        )

    def test_json_in_a_markdown_fence_is_still_parsed(self) -> None:
        raw = '```json\n{"category": "earnings", "insight": "Raport kwartalny."}\n```'

        assert parse_mistake_diagnosis(raw) == MistakeDiagnosis(
            MistakeCategory.EARNINGS, "Raport kwartalny."
        )

    def test_plain_text_falls_back_to_other_with_the_whole_text(self) -> None:
        raw = "  Zignorowałem szerszy kontekst makro.  "

        assert parse_mistake_diagnosis(raw) == MistakeDiagnosis(
            MistakeCategory.OTHER, "Zignorowałem szerszy kontekst makro."
        )

    def test_unknown_category_is_other_but_keeps_the_insight(self) -> None:
        raw = '{"category": "ASTROLOGY", "insight": "Pełnia księżyca."}'

        assert parse_mistake_diagnosis(raw) == MistakeDiagnosis(
            MistakeCategory.OTHER, "Pełnia księżyca."
        )

    @pytest.mark.parametrize(
        "raw",
        ['{"category": "MACRO"}', '{"category": "MACRO", "insight": "  "}', "[1, 2]"],
    )
    def test_json_without_an_insight_keeps_the_raw_answer(self, raw: str) -> None:
        diagnosis = parse_mistake_diagnosis(raw)

        assert diagnosis.insight == raw
        assert diagnosis.category in (MistakeCategory.MACRO, MistakeCategory.OTHER)

    def test_broken_json_is_other(self) -> None:
        raw = '{"category": "MACRO", "insight": "urwane'

        assert parse_mistake_diagnosis(raw) == MistakeDiagnosis(MistakeCategory.OTHER, raw)


class TestCategories:
    def test_stored_values_parse_case_insensitively(self) -> None:
        assert parse_mistake_category(" sector_rotation ") is MistakeCategory.SECTOR_ROTATION
        assert parse_mistake_category(None) is None
        assert parse_mistake_category("UNKNOWN") is None

    def test_count_skips_missing_categories(self) -> None:
        counts = count_categories(
            [MistakeCategory.MACRO, None, MistakeCategory.MACRO, MistakeCategory.EARNINGS]
        )

        assert counts == {MistakeCategory.MACRO: 2, MistakeCategory.EARNINGS: 1}
        assert list(counts) == [MistakeCategory.MACRO, MistakeCategory.EARNINGS]

    def test_every_category_has_a_polish_label(self) -> None:
        assert set(MISTAKE_CATEGORY_LABELS_PL) == set(MistakeCategory)


class TestParseBatchDiagnoses:
    """EXP-17: jedna odpowiedź LLM nad wszystkimi pudłami tygodnia → diagnoza per pudło."""

    def test_entries_map_to_their_positions(self) -> None:
        raw = json.dumps(
            {
                "diagnoses": [
                    {"n": 2, "category": "earnings", "insight": " Raport. "},
                    {"n": 1, "category": "MACRO", "insight": "Stopy."},
                ]
            }
        )

        assert parse_batch_diagnoses(raw, 2) == (
            MistakeDiagnosis(MistakeCategory.MACRO, "Stopy."),
            MistakeDiagnosis(MistakeCategory.EARNINGS, "Raport."),
        )

    def test_missing_unusable_and_out_of_range_entries_leave_none(self) -> None:
        raw = json.dumps(
            {
                "diagnoses": [
                    "not an object",
                    {"n": "1", "insight": "pozycja jako tekst"},
                    {"n": True, "insight": "bool to nie pozycja"},
                    {"n": 2, "insight": "  "},
                    {"n": 3, "category": "???", "insight": "Inne."},
                    {"n": 3, "category": "NOISE", "insight": "Duplikat."},
                    {"n": 9, "insight": "Poza zakresem."},
                ]
            }
        )

        assert parse_batch_diagnoses(raw, 3) == (
            None,
            None,
            MistakeDiagnosis(MistakeCategory.OTHER, "Inne."),
        )

    def test_answer_without_a_list_diagnoses_nothing(self) -> None:
        # Bez listy `diagnoses` nie wiadomo, którego pudła dotyczy tekst — żadna
        # pozycja nie dostaje diagnozy, placeholdery zostają.
        assert parse_batch_diagnoses("Makro.", 2) == (None, None)
        assert parse_batch_diagnoses('{"diagnoses": "x", "insight": "Makro."}', 1) == (None,)

    def test_empty_answer_diagnoses_nothing(self) -> None:
        assert parse_batch_diagnoses("   ", 2) == (None, None)
