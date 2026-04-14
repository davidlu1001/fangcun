import pytest
from core import SealGenerator


@pytest.fixture(scope="session")
def gen() -> SealGenerator:
    """Shared SealGenerator instance (warm cache across tests).

    Explicit no_api_cache=False: regression suite needs cache hits for reproducibility.
    """
    return SealGenerator(no_api_cache=False)
