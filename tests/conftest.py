"""Top-level pytest fixtures shared across unit + regression tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_numpy_random() -> None:
    """Snapshot/restore numpy's global RNG around each test.

    Texture tests call np.random.seed() explicitly. Without this fixture, that
    seeded state would leak into the next test and quietly make non-deterministic
    behaviour deterministic — masking real bugs.
    """
    state = np.random.get_state()
    yield
    np.random.set_state(state)
