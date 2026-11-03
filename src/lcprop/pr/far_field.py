"""Shared PR output-plane presentation; canonical transform lives in optics."""
import numpy as np
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.products.data_model import make_field


def far_field_product(intensity, s_x, s_y, *, observation="accepted output plane"):
    return make_field(
        "far_field_intensity", "Output Far-Field Intensity", intensity,
        ("s_x", "s_y"), "far_field_intensity", {"s_x": "1", "s_y": "1"},
        quantity="direction_cosine_field_norm_density",
        value_unit="normalized field norm / direction-cosine²", colormap="magma",
        coordinates={"s_x": s_x, "s_y": s_y, "observation": observation,
                     "normalization": "fftshift(fft2(A))*dx*dy; squared modulus times (n/lambda0)^2",
                     "qualification": "coherence-aware field-norm density, not physical mW; no window or carrier recentering"},
    )


def completed_far_field(result):
    launch = result.launch_summary
    wavelengths = launch.get("wavelengths_um", ())
    n = launch.get("refractive_index")
    if not wavelengths or n is None:
        profile = getattr(result, "resolved_profile", {})
        channels = profile.get("beam_request", {}).get("channels", ())
        wavelengths = [channel["wavelength_um"] for channel in channels]
        n = profile.get("material", {}).get("refractive_index")
    if not wavelengths or n is None:
        return None  # Older incomplete result metadata cannot define angular axes.
    spectrum = direction_cosine_spectrum(
        np.asarray(result.A_final), dx_um=float(result.grid_summary["dx_um"]),
        dy_um=float(result.grid_summary["dy_um"]), wavelength_um=float(wavelengths[0]),
        refractive_index=float(n), coherence_groups=tuple(launch["coherence_groups"]),
    )
    fallback = result.diagnostics.get("final_optical_observation") == "unpropagated_launch_fallback"
    return far_field_product(spectrum.intensity, spectrum.s_x, spectrum.s_y,
                             observation="unpropagated launch fallback" if fallback else "final accepted output plane")
