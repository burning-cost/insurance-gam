"""
Tests that importing GAM subpackages without their extras installed raises
a clear, actionable ImportError rather than a confusing traceback.

We simulate missing dependencies by temporarily removing them from sys.modules
and blocking re-import via a custom meta path finder.
"""

from __future__ import annotations

import importlib
import sys
import types
import pytest


class _BlockImport:
    """Meta path finder that blocks import of a named module."""

    def __init__(self, blocked: str):
        self._blocked = blocked

    def find_module(self, name, path=None):
        if name == self._blocked or name.startswith(self._blocked + "."):
            return self

    def load_module(self, name):
        raise ImportError(f"Simulated missing dependency: {name}")


def _reload_subpackage_without(subpackage: str, blocked_dep: str):
    """
    Remove *subpackage* and its children from sys.modules, block *blocked_dep*,
    then attempt to import *subpackage*. Returns the ImportError if raised.
    """
    # Remove the subpackage from the module cache so it reimports fresh
    to_remove = [k for k in sys.modules if k == subpackage or k.startswith(subpackage + ".")]
    saved = {k: sys.modules.pop(k) for k in to_remove}

    blocker = _BlockImport(blocked_dep)
    sys.meta_path.insert(0, blocker)

    try:
        importlib.import_module(subpackage)
        return None  # No error raised — test will fail
    except ImportError as exc:
        return exc
    finally:
        sys.meta_path.remove(blocker)
        # Restore previously loaded modules so other tests are unaffected
        sys.modules.update(saved)


def test_ebm_missing_interpret_gives_helpful_error():
    err = _reload_subpackage_without("insurance_gam.ebm", "interpret")
    assert err is not None, "Expected ImportError when interpret is missing"
    msg = str(err)
    assert "ebm" in msg.lower()
    assert "pip install insurance-gam[ebm]" in msg


def test_anam_missing_torch_gives_helpful_error():
    err = _reload_subpackage_without("insurance_gam.anam", "torch")
    assert err is not None, "Expected ImportError when torch is missing"
    msg = str(err)
    assert "anam" in msg.lower()
    assert "pip install insurance-gam[anam]" in msg


def test_pin_missing_torch_gives_helpful_error():
    err = _reload_subpackage_without("insurance_gam.pin", "torch")
    assert err is not None, "Expected ImportError when torch is missing"
    msg = str(err)
    assert "pin" in msg.lower()
    assert "pip install insurance-gam[pin]" in msg
