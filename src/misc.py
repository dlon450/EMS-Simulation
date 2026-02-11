from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import random

import io as io_

from .defs import EventForm, Priority, PRIORITIES
from .events import Event
from .distributions import ParsedDistribution, parse_distribution_spec


@dataclass
class DistrRng:
    """Distribution + RNG wrapper (Julia-compatible intent).

    Julia uses ``DistrRng`` to bundle a distribution object with a seeded
    ``MersenneTwister`` so that sampling is deterministic and isolated.

    In this Python port, input files store the distribution as a string
    expression (e.g. ``"Exponential(0.5)"``). We parse a safe subset of
    those expressions (see :mod:`jemss.distributions`) and sample using an
    independent :class:`random.Random` instance.

    Notes
    -----
    * If ``seed`` is ``None`` or negative, a random 32-bit seed is chosen.
    * ``distribution`` may be ``None`` -> treated as ``Constant(0)``.
    """

    distribution: Any = None  # string spec or numeric constant
    seed: Optional[int] = None
    # Optional RNG used to generate a seed when ``seed`` is negative.
    # This is useful for end-to-end determinism when the overall simulation
    # is seeded.
    seed_rng: Optional[random.Random] = None

    # internal state
    _rng: random.Random = field(init=False, repr=False)
    _parsed: ParsedDistribution = field(init=False, repr=False)

    def __post_init__(self) -> None:
        seed = self.seed
        if seed is None:
            seed = -1
        if seed < 0:
            src = self.seed_rng
            # Both ``random.Random`` and the ``random`` module provide
            # ``getrandbits``.
            if src is None:
                seed = random.getrandbits(32)
            else:
                seed = int(src.getrandbits(32))
        self._rng = random.Random(int(seed))
        self._parsed = parse_distribution_spec(self.distribution)

    def sample(self) -> float:
        """Draw one sample."""
        return float(self._parsed.sample(self._rng))

    def mean(self) -> Optional[float]:
        """Return the mean if available for the parsed distribution."""
        return self._parsed.mean()


class XMLElement:
    """Placeholder for an XML element object.

    The Julia implementation uses EzXML.jl. In Python we'll likely use
    ``xml.etree.ElementTree`` or ``lxml`` later.
    """


class Histogram:
    """Placeholder for a histogram type.

    In Julia the project uses StatsBase.Histogram. We'll introduce a
    real histogram implementation later (or use numpy/pandas) once the
    simulation core is working.
    """


@dataclass
class File:
    name: str = ""
    path: str = ""
    stream: io_.IOBase = field(default_factory=io_.StringIO)
    checksum: int = 0


@dataclass
class EventsFile:
    io: io_.IOBase = field(default_factory=io_.StringIO)
    event_filter: Dict[EventForm, bool] = field(
        default_factory=lambda: {e: True for e in EventForm}
    )


@dataclass
class Resimulation:
    use: bool = False
    time_tolerance: float = 0.0
    events: List[Event] = field(default_factory=list)
    events_children: List[List[Event]] = field(default_factory=list)
    prev_event_index: Optional[int] = None
    event_filter: Dict[EventForm, bool] = field(default_factory=dict)
    do_dispatch: bool = False
    do_move_up: bool = False


@dataclass
class MobilisationDelay:
    use: bool = False
    distr_rng: Optional[DistrRng] = None
    expected_duration: float = 0.0


@dataclass
class Redispatch:
    allow: bool = False
    # conditions[from_priority][to_priority] -> bool
    conditions: Dict[Priority, Dict[Priority, bool]] = field(
        default_factory=lambda: {p1: {p2: False for p2 in PRIORITIES} for p1 in PRIORITIES}
    )

    @classmethod
    def default(cls) -> "Redispatch":
        """Default redispatch policy: only to higher priority."""
        r = cls(allow=True)
        for p1 in PRIORITIES:
            for p2 in PRIORITIES:
                if p2 == Priority.HIGH and p1 != Priority.HIGH:
                    r.conditions[p1][p2] = True
        return r

    def can_redispatch(self, from_priority: Priority, to_priority: Priority) -> bool:
        if not self.allow:
            return False
        return self.conditions.get(from_priority, {}).get(to_priority, False)


Distribution = Any
Sampleable = Any