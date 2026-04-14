"""P3#13: version is sourced from pyproject.toml (single source of truth)."""

import re

import pytest


@pytest.mark.unit
class TestVersion:
    def test_core_exposes_version(self) -> None:
        import core
        assert hasattr(core, "__version__")
        # PEP 440-ish: at minimum N.N.N format
        assert re.match(r"^\d+\.\d+\.\d+", core.__version__), core.__version__

    def test_version_matches_pyproject(self) -> None:
        """The exposed __version__ should track pyproject.toml exactly.

        If the git tag is ahead of pyproject (as it once was after v1.1.0),
        this test catches the drift.
        """
        import tomllib
        from pathlib import Path

        import core

        pyproject = Path(core.__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as f:
            expected = tomllib.load(f)["project"]["version"]
        assert core.__version__ == expected
