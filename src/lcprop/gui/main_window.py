"""Compatibility alias for the canonical LC application window."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("lcprop.lc.gui.main_window")
_sys.modules[__name__] = _canonical
