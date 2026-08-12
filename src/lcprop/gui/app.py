"""Compatibility launcher for the canonical LC application entry point."""

from lcprop.lc.gui.app import main


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
