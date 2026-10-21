"""NIEZ-32: krok kopii tabel dowodowych w Slow Loopie i jego workflow.

Kopia jest szyfrowana `age` kluczem odbiorcy z sekretu `BACKUP_AGE_RECIPIENT`
PRZED zapisem; bez sekretu krok jest pomijany z WARNING-iem — nieszyfrowany
upload nie powstaje nigdy. Workflow uploaduje wyłącznie `*.age` (retencja 90 dni).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import main_trainer
from src.application.ports import ArchiveReceipt, EvidenceSourcePort
from src.config import NON_RECIPE_SETTINGS_FIELDS, Settings
from src.infrastructure.adapters.age_archive import is_age_recipient

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"
RECIPIENT = "age1" + "q" * 58
NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
SECRET_REF = "${{ secrets.BACKUP_AGE_RECIPIENT }}"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
    )


class _Source(EvidenceSourcePort):
    def read_rows(self, table: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        return [{"id": 1, "commitment_salt": "salt"}]


def test_without_recipient_the_step_is_skipped_with_a_warning(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    repository = mocker.patch("main_trainer.SupabaseRepository")

    main_trainer._run_evidence_export(settings, now=NOW, out_dir=tmp_path)

    repository.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert any(
        r.levelname == "WARNING" and "BACKUP_AGE_RECIPIENT" in r.getMessage()
        for r in caplog.records
    )


def test_placeholder_recipient_is_skipped_too(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    settings.backup_age_recipient = "age1replace-with-your-age-public-key"
    repository = mocker.patch("main_trainer.SupabaseRepository")

    main_trainer._run_evidence_export(settings, now=NOW, out_dir=tmp_path)

    repository.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_export_is_encrypted_and_its_sha256_logged(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    import logging

    settings.backup_age_recipient = RECIPIENT
    mocker.patch("main_trainer.SupabaseRepository", return_value=_Source())
    archive_cls = mocker.patch("main_trainer.AgeEncryptedArchive")
    archive_cls.return_value.store.return_value = ArchiveReceipt(
        path=str(tmp_path / "evidence-2026-W43.tar.gz.age"), sha256="cd" * 32, size_bytes=99
    )

    with caplog.at_level(logging.INFO, logger="main_trainer"):
        main_trainer._run_evidence_export(settings, now=NOW, out_dir=tmp_path)

    archive_cls.assert_called_once_with(RECIPIENT, tmp_path)
    name, files = archive_cls.return_value.store.call_args.args
    assert name == "evidence-2026-W43"
    assert b"salt" not in b"".join(files.values())
    assert "sha256=" + "cd" * 32 in caplog.text


def test_failed_export_does_not_break_the_run(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    settings.backup_age_recipient = RECIPIENT
    mocker.patch("main_trainer.SupabaseRepository", return_value=_Source())
    archive_cls = mocker.patch("main_trainer.AgeEncryptedArchive")
    archive_cls.return_value.store.side_effect = FileNotFoundError("age")

    main_trainer._run_evidence_export(settings, now=NOW, out_dir=tmp_path)

    assert "Evidence export failed" in caplog.text


def test_recipient_is_outside_the_recipe_hash() -> None:
    # Klucz publiczny nic nie odszyfruje; poza przepisem, bo nie zmienia prognozy.
    assert "backup_age_recipient" in NON_RECIPE_SETTINGS_FIELDS


def _steps() -> list[dict[str, Any]]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return [step for job in spec["jobs"].values() for step in job.get("steps", [])]


def _trainer_index(steps: list[dict[str, Any]]) -> int:
    return next(i for i, s in enumerate(steps) if str(s.get("run", "")).endswith("main_trainer.py"))


def test_trainer_step_maps_the_recipient_secret() -> None:
    steps = _steps()

    assert steps[_trainer_index(steps)]["env"]["BACKUP_AGE_RECIPIENT"] == SECRET_REF


def test_age_is_installed_before_training_without_blocking_it() -> None:
    steps = _steps()
    install = next(i for i, s in enumerate(steps) if "install -y -qq age" in str(s.get("run", "")))

    assert install < _trainer_index(steps)
    assert steps[install]["continue-on-error"] is True
    assert steps[install]["env"]["BACKUP_AGE_RECIPIENT"] == SECRET_REF
    assert '[ -n "$BACKUP_AGE_RECIPIENT" ]' in steps[install]["run"]


def test_only_encrypted_files_are_uploaded_with_90_day_retention() -> None:
    steps = _steps()
    uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]

    (upload,) = uploads
    paths = [p.strip() for p in str(upload["with"]["path"]).splitlines() if p.strip()]
    assert paths and all(p.endswith(".age") for p in paths)
    assert upload["with"]["retention-days"] == 90
    assert upload["with"]["if-no-files-found"] == "ignore"
    assert str(upload.get("if", "")).replace(" ", "") == "${{!cancelled()}}"
    assert steps.index(upload) > _trainer_index(steps)


def test_guard_rejects_a_plaintext_upload_path() -> None:
    upload = {"with": {"path": "data/backup/\n"}}
    paths = [p.strip() for p in str(upload["with"]["path"]).splitlines() if p.strip()]

    assert not all(p.endswith(".age") for p in paths)


def test_env_example_ships_a_recipient_placeholder_that_is_not_a_key() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1] for line in lines if line.startswith("BACKUP_AGE_RECIPIENT=")]

    assert values == ["age1replace-with-your-age-public-key"]
    assert not is_age_recipient(values[0])


def test_backup_directory_is_git_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/data/backup/" in ignored
