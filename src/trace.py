from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import hashlib
import struct

from .events import Event


# We use Julia-style sentinel values in the trace (nullIndex=-1, nullTime=-1.0)
_NULL_INDEX = -1
_NULL_TIME = -1.0


def _idx(x: Optional[int]) -> int:
    return _NULL_INDEX if x is None else int(x)


def _time(x: Optional[float]) -> float:
    return _NULL_TIME if x is None else float(x)


def event_row(event: Event) -> Tuple[int, int, int, float, int, int, int]:
    """Return a stable tuple representation of *event*.

    The tuple is:

    ``(index, parent_index, form_value, time, amb_index, call_index, station_index)``.
    """

    return (
        _idx(event.index),
        _idx(event.parent_index),
        int(event.form.value),
        _time(event.time),
        _idx(event.amb_index),
        _idx(event.call_index),
        _idx(event.station_index),
    )


@dataclass
class TraceRecorder:
    """Record executed events and optionally compute a digest."""

    store_events: bool = False
    compute_digest: bool = True

    events: List[Tuple[int, int, int, float, int, int, int]] = field(default_factory=list)
    _hasher: Optional[hashlib._Hash] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.compute_digest:
            self._hasher = hashlib.sha256()

    def record(self, event: Event) -> None:
        row = event_row(event)

        if self.store_events:
            self.events.append(row)

        if self._hasher is not None:
            # Pack as big-endian: 3 int32, float64, 3 int32.
            # This is stable across platforms.
            b = struct.pack(">iiidiii", row[0], row[1], row[2], row[3], row[4], row[5], row[6])
            self._hasher.update(b)

    def digest_hex(self) -> Optional[str]:
        if self._hasher is None:
            return None
        return self._hasher.hexdigest()


def digest_events(events: Sequence[Event]) -> str:
    """Compute a digest for an iterable of :class:`~jemss.events.Event`."""

    tr = TraceRecorder(store_events=False, compute_digest=True)
    for e in events:
        tr.record(e)
    d = tr.digest_hex()
    assert d is not None
    return d
