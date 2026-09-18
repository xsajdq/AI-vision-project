from collections import deque
from datetime import datetime, timezone
from itertools import count
from threading import Lock

from .config import settings
from .schemas import HistoryEntry, InspectionKind

_lock = Lock()
_entries: deque[HistoryEntry] = deque(maxlen=settings.history_limit)
_next_id = count(1)


def record(kind: InspectionKind, filename: str, verdict: str, anomaly_score: float, threshold: float) -> HistoryEntry:
    entry = HistoryEntry(
        id=next(_next_id),
        kind=kind,
        filename=filename,
        verdict=verdict,
        anomaly_score=anomaly_score,
        threshold=threshold,
        timestamp=datetime.now(timezone.utc),
    )
    with _lock:
        _entries.appendleft(entry)
    return entry


def list_recent() -> list[HistoryEntry]:
    with _lock:
        return list(_entries)
