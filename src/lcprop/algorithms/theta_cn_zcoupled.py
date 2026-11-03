"""Compatibility alias for :mod:`lcprop.lc.theta_cn_zcoupled`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("lcprop.lc.theta_cn_zcoupled")
_sys.modules[__name__] = _canonical
