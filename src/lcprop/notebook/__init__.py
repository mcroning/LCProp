"""Optional notebook interfaces, loaded only when explicitly requested."""

__all__ = ["InteractiveLC"]


def __getattr__(name):
    if name == "InteractiveLC":
        from .interactive_lc import InteractiveLC

        globals()[name] = InteractiveLC
        return InteractiveLC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
