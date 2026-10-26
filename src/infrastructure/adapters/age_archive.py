"""NIEZ-32: magazyn kopii zapasowej szyfrowany `age` (https://age-encryption.org).

Artefakt GitHub Actions w publicznym repo może pobrać każdy zalogowany, a eksport
track recordu bez atestacji byłby de facto jego publikacją. Dlatego jawny tar.gz
istnieje wyłącznie w pamięci: idzie przez stdin do `age --encrypt --recipient`
(klucz publiczny odbiorcy z sekretu `BACKUP_AGE_RECIPIENT`), a na dysk trafia
tylko wynik z nagłówkiem age. Wyjście bez nagłówka nie jest zapisywane wcale.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import subprocess
import tarfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.application.ports import ArchiveReceipt, EvidenceArchivePort

AGE_HEADER = b"age-encryption.org/v1"
ARCHIVE_SUFFIX = ".tar.gz.age"
AGE_TIMEOUT_S = 120
# Odbiorca X25519 (bech32, „age1” + 58 znaków) albo klucz SSH, który age przyjmuje.
_RECIPIENT = re.compile(r"^(age1[02-9ac-hj-np-z]{58}|ssh-(ed25519|rsa) \S+)$")

Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class AgeEncryptionError(RuntimeError):
    """`age` nie zaszyfrował danych — kopia nie powstaje, nic nie trafia na dysk."""


def is_age_recipient(value: str) -> bool:
    """Czy `value` wygląda na klucz publiczny odbiorcy age (nie tożsamość, nie placeholder)."""
    return bool(_RECIPIENT.match(value.strip()))


def _tar_gz(files: Mapping[str, bytes]) -> bytes:
    """Deterministyczny tar.gz w pamięci (stałe mtime, kolejność nazw)."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for name in sorted(files):
            content = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class AgeEncryptedArchive(EvidenceArchivePort):
    """`store` → `<out_dir>/<name>.tar.gz.age`; `run` to `subprocess.run` (testy podmieniają)."""

    def __init__(
        self,
        recipient: str,
        out_dir: Path,
        *,
        age_binary: str = "age",
        run: Runner = subprocess.run,
    ) -> None:
        if not is_age_recipient(recipient):
            raise ValueError("BACKUP_AGE_RECIPIENT is not an age recipient public key")
        self._recipient = recipient.strip()
        self._out_dir = out_dir
        self._binary = age_binary
        self._run = run

    def store(self, name: str, files: Mapping[str, bytes]) -> ArchiveReceipt:
        ciphertext = self._encrypt(_tar_gz(files))
        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / f"{name}{ARCHIVE_SUFFIX}"
        path.write_bytes(ciphertext)
        return ArchiveReceipt(
            path=str(path),
            sha256=hashlib.sha256(ciphertext).hexdigest(),
            size_bytes=len(ciphertext),
        )

    def _encrypt(self, plaintext: bytes) -> bytes:
        kwargs: dict[str, Any] = {
            "input": plaintext,
            "capture_output": True,
            "check": False,
            "timeout": AGE_TIMEOUT_S,
        }
        completed = self._run([self._binary, "--encrypt", "--recipient", self._recipient], **kwargs)
        if completed.returncode != 0:
            raise AgeEncryptionError(f"age exited with code {completed.returncode}")
        if not completed.stdout.startswith(AGE_HEADER):
            raise AgeEncryptionError("age output has no age header — refusing to write it")
        return completed.stdout
