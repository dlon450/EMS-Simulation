from __future__ import annotations

import random

from .defs import EventForm
from .misc import DistrRng
from .simulator import Simulation


def test_event_ordering() -> None:
    """Verify reference-compatible event ordering + tie-break."""

    sim = Simulation()

    # Two events at the same time: the later inserted should execute first.
    e1 = sim.add_event(form=EventForm.CALL_ARRIVES, time=1.0, add_event_to_amb=False)
    e2 = sim.add_event(form=EventForm.CONSIDER_DISPATCH, time=1.0, add_event_to_amb=False)

    n1 = sim.next_event()
    n2 = sim.next_event()
    assert n1 is e2, "Same-time tie-break failed: newer event should execute first"
    assert n2 is e1

    # Different times: smallest time should execute first.
    sim = Simulation()
    a = sim.add_event(form=EventForm.CALL_ARRIVES, time=2.0, add_event_to_amb=False)
    b = sim.add_event(form=EventForm.CALL_ARRIVES, time=0.5, add_event_to_amb=False)
    c = sim.add_event(form=EventForm.CALL_ARRIVES, time=1.0, add_event_to_amb=False)

    assert sim.next_event() is b
    assert sim.next_event() is c
    assert sim.next_event() is a


def test_distr_rng_seed_reproducible() -> None:
    """If seed=-1 and the seed RNG is fixed, sampling should be reproducible."""

    seed_src_1 = random.Random(123)
    seed_src_2 = random.Random(123)

    d1 = DistrRng(distribution="Normal(0, 1)", seed=-1, seed_rng=seed_src_1)
    d2 = DistrRng(distribution="Normal(0, 1)", seed=-1, seed_rng=seed_src_2)

    xs1 = [d1.sample() for _ in range(5)]
    xs2 = [d2.sample() for _ in range(5)]
    assert xs1 == xs2, "DistrRng not reproducible when seed_rng is fixed"


def run_all() -> None:
    test_event_ordering()
    test_distr_rng_seed_reproducible()
