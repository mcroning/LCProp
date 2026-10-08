"""Matplotlib-only labels; never convert stored units or scientific values."""
import re

_GREEK = {'ψ': r'\psi', 'τ': r'\tau', 'γ': r'\gamma', 'λ': r'\lambda',
          'θ': r'\theta', 'φ': r'\phi', 'Δ': r'\Delta', 'ε': r'\epsilon',
          'μ': r'\mu', 'µ': r'\mu'}


def scientific_text(text):
    """Format plain scientific glyphs while preserving existing mathtext spans."""
    parts = re.split(r'(\$[^$]*\$)', str(text))
    for i in range(0, len(parts), 2):
        value = parts[i]
        value = re.sub(r'(?<![A-Za-z])(?:um|µm|μm)(?:\^2|²)?',
                       lambda m: r'$\mu\mathrm{m}' + ('^{2}' if m[0].endswith(('^2', '²')) else '') + '$', value)
        # Do not transform the mathtext just inserted above.
        chunks = re.split(r'(\$[^$]*\$)', value)
        for j in range(0, len(chunks), 2):
            chunks[j] = ''.join('$'+_GREEK[c]+'$' if c in _GREEK else
                                '$^{2}$' if c == '²' else c for c in chunks[j])
        parts[i] = ''.join(chunks)
    return ''.join(parts)


def unit_label(unit):
    unit = str(unit)
    prefix = ''
    if unit.startswith('log10 '):
        prefix, unit = r'$\log_{10}$ ', unit[6:]
    if unit in ('1/um^2', '1/µm²', '1/μm²', '1/um²'):
        return prefix + r'$1/(\mu\mathrm{m})^{2}$'
    return prefix + scientific_text(unit)


def axis_label(label, units):
    symbol = {'psi': r'$\psi$', 'tau': r'$\tau$',
              's_x': r'$s_x$', 's_y': r'$s_y$',
              'E_x': r'$E_x$', 'E_y': r'$E_y$'}.get(label, scientific_text(label))
    unit = units.get(label)
    return f'{symbol} ({unit_label(unit)})' if unit else symbol
