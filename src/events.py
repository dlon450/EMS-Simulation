"""Event types.

This module exists to avoid circular imports:

* ``Simulation`` needs the :class:`~jemss.events.Event` type to manage the
  future-event list.
* Entities such as :class:`~jemss.entities.Ambulance` store their next event.

By placing ``Event`` in its own module, both layers can import it without
depending on each other.
"""

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
