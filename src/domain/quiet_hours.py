"""ALR-09 — cisza nocna pushy: okno godzin, w którym push dociera bez dźwięku.

Watch szoku chodzi 24/7 (BTC/ETH), więc szok krypto o 3:00 budziłby odbiorcę.
Całkowite tłumienie gubiłoby informację, dlatego w oknie ciszy kanał push
dostarcza wiadomość po cichu, a nie wcale. Okno liczy się w strefie odbiorcy
(`display_timezone`) i może przechodzić przez północ (22:00–07:00).

Czysta funkcja, tylko stdlib — strefę czasu rozwiązuje `zoneinfo`.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

DEFAULT_QUIET_START = time(22, 0)
DEFAULT_QUIET_END = time(7, 0)


def is_quiet(
    now_utc: datetime,
    tz: str,
    start: time = DEFAULT_QUIET_START,
    end: time = DEFAULT_QUIET_END,
) -> bool:
    """Czy chwila `now_utc` wypada w oknie ciszy [start, end) w strefie `tz`.

    Początek okna należy do ciszy, koniec już nie (07:00 to pora pobudki).
    `start > end` oznacza okno przez północ; `start == end` — okno puste, cisza
    nigdy. Naiwny `datetime` czytamy jako UTC, tak jak zapisuje czas całe repo.
    """
    if start == end:
        return False
    aware = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=UTC)
    local = aware.astimezone(ZoneInfo(tz)).time().replace(tzinfo=None)
    if start < end:
        return start <= local < end
    return local >= start or local < end
