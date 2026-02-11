from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .defs import EventForm


@dataclass
class Event:
    """A discrete event in the simulation timeline."""

    index: Optional[int] = None
    parent_index: Optional[int] = None
    form: EventForm = EventForm.NULL
    time: Optional[float] = None
    amb_index: Optional[int] = None
    call_index: Optional[int] = None
    station_index: Optional[int] = None
