"""Display-only numeric text; no conversion of stored scientific values."""
from numbers import Real


def format_number(value, *, quantity="ordinary", exact=False):
    if value is None:
        return "Unavailable"
    if isinstance(value, bool) or not isinstance(value, Real):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if exact or "overlap" in quantity:
        text = repr(float(value))
    else:
        # Coordinates have display-scale precision; diagnostics retain more detail.
        digits = 6 if quantity == "coordinate" else 8 if quantity == "tolerance" else 6
        text = format(float(value), f".{digits}g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text
