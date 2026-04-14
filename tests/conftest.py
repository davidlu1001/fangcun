import pytest
from core import SealGenerator


@pytest.fixture(scope="session")
def gen():
    """Shared SealGenerator instance (warm cache across tests)."""
    return SealGenerator(no_api_cache=False)
