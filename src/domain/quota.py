"""Domena Quota Monitoring — alerty o wyczerpaniu limitów subskrypcji.

Cel: żadne ciche wyczerpanie nie ma prawa zostać niezauważone w cyklu.
Adaptery (Alpha Vantage, OpenAI, Finnhub, Resend) emitują `QuotaAlert`,
a raport mailowy wystawia banner na samej górze.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class QuotaSeverity(Enum):
    """Poziom alertu:

    - `WARNING`  — degradacja, ale system działa (np. rotacja kluczy
      sukcesem). Banner żółty, brak prefiksu subjectu.
    - `CRITICAL` — operacja nie powiodła się i dane są stracone w cyklu
      (np. wszystkie klucze AV exhausted, email nie poszedł). Banner
      czerwony, subject prefix `⚠️ [QUOTA]`.
    """

    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Numeryczna ranga do porównań (max severity, sortowanie)."""
        return {QuotaSeverity.WARNING: 1, QuotaSeverity.CRITICAL: 2}[self]


@dataclass(frozen=True)
class QuotaAlert:
    source: str            # "Alpha Vantage" / "OpenAI" / "Finnhub" / "Resend"
    severity: QuotaSeverity
    message: str           # co się stało, krótko
    action: str            # konkretny krok do podjęcia przez właściciela
    occurred_at: datetime
