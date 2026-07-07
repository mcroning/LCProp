from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


def spin_box(lo: int, hi: int, value: int) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    return w


def double_spin_box(lo: float, hi: float, value: float, *, decimals: int = 6) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setValue(value)
    return w
