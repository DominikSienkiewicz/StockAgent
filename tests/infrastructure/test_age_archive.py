"""NIEZ-32: magazyn kopii szyfrujący `age` kluczem odbiorcy PRZED zapisem.

Artefakt GitHub Actions w publicznym repo pobierze każdy zalogowany, więc na dysk
(i do uploadu) trafia wyłącznie szyfrogram. Jawny tar idzie do `age` przez stdin.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from src.infrastructure.adapters.age_archive import (
    AGE_HEADER,
    AgeEncryptedArchive,
    AgeEncryptionError,
    is_age_recipient,
)

RECIPIENT = "age1" + "q" * 58
FILES = {"prediction_logs.csv": b"id,symbol\np1,NVDA\n", "manifest.json": b"{}"}


class _FakeAge:
    def __init__(self, *, returncode: int = 0, stdout: bytes | None = None) -> None:
        self.calls: list[tuple[list[str], bytes]] = []
        self._returncode = returncode
        self._stdout = stdout

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((args, kwargs["input"]))
        stdout = self._stdout if self._stdout is not None else AGE_HEADER + b"\nciphertext"
        return subprocess.CompletedProcess(args, self._returncode, stdout=stdout, stderr=b"")


def test_store_writes_only_the_ciphertext_and_reports_its_sha256(tmp_path: Path) -> None:
    fake = _FakeAge()
    archive = AgeEncryptedArchive(RECIPIENT, tmp_path, run=fake)

    receipt = archive.store("evidence-2026-W43", FILES)

    written = tmp_path / "evidence-2026-W43.tar.gz.age"
    assert [p.name for p in tmp_path.iterdir()] == [written.name]
    ciphertext = written.read_bytes()
    assert ciphertext.startswith(AGE_HEADER)
    assert receipt.path == str(written)
    assert receipt.sha256 == hashlib.sha256(ciphertext).hexdigest()
    assert receipt.size_bytes == len(ciphertext)


def test_age_gets_the_recipient_and_the_tarball_on_stdin(tmp_path: Path) -> None:
    fake = _FakeAge()

    AgeEncryptedArchive(RECIPIENT, tmp_path, run=fake).store("evidence-2026-W43", FILES)

    ((args, plaintext),) = fake.calls
    assert args == ["age", "--encrypt", "--recipient", RECIPIENT]
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == sorted(FILES)
        member = tar.extractfile("prediction_logs.csv")
        assert member is not None and member.read() == FILES["prediction_logs.csv"]


def test_failed_encryption_writes_nothing(tmp_path: Path) -> None:
    archive = AgeEncryptedArchive(RECIPIENT, tmp_path, run=_FakeAge(returncode=1))

    with pytest.raises(AgeEncryptionError):
        archive.store("evidence-2026-W43", FILES)

    assert list(tmp_path.iterdir()) == []


def test_output_that_is_not_an_age_file_is_never_written(tmp_path: Path) -> None:
    # Strażnik „nigdy nieszyfrowany upload”: bez nagłówka age nic nie trafia na dysk.
    archive = AgeEncryptedArchive(RECIPIENT, tmp_path, run=_FakeAge(stdout=b"id,symbol\n"))

    with pytest.raises(AgeEncryptionError):
        archive.store("evidence-2026-W43", FILES)

    assert list(tmp_path.iterdir()) == []


def test_missing_age_binary_writes_nothing(tmp_path: Path) -> None:
    def no_binary(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(args[0])

    with pytest.raises(FileNotFoundError):
        AgeEncryptedArchive(RECIPIENT, tmp_path, run=no_binary).store("x", FILES)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (RECIPIENT, True),
        ("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG", True),
        ("age1replace-with-your-age-public-key", False),
        ("", False),
        ("AGE-SECRET-KEY-1QQQ", False),
    ],
)
def test_recipient_shape(value: str, expected: bool) -> None:
    assert is_age_recipient(value) is expected


def test_invalid_recipient_is_rejected_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AgeEncryptedArchive("AGE-SECRET-KEY-1QQQ", tmp_path, run=_FakeAge())


@pytest.mark.skipif(shutil.which("age-keygen") is None, reason="age not installed")
def test_round_trip_with_the_real_age_binary(tmp_path: Path) -> None:
    identity = tmp_path / "key.txt"
    subprocess.run(["age-keygen", "-o", str(identity)], check=True, capture_output=True)
    recipient = next(
        line.split(": ", 1)[1]
        for line in identity.read_text().splitlines()
        if line.startswith("# public key: ")
    )
    out = tmp_path / "out"

    receipt = AgeEncryptedArchive(recipient, out).store("evidence-2026-W43", FILES)

    plaintext = subprocess.run(
        ["age", "--decrypt", "-i", str(identity), receipt.path], check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == sorted(FILES)
