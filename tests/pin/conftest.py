"""conftest.py — Skip all PIN tests if torch is not installed."""

import pytest

pytest.importorskip("torch", reason="torch not installed; skip PIN tests")
