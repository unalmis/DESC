"""Compute functions for turbulent transport.

References
----------
.. [1] J. H. E. Proll et al., "TEM turbulence optimisation in stellarators,"
       Plasma Phys. Control. Fusion 58, 014006 (2016).
       https://doi.org/10.1088/0741-3335/58/1/014006.
.. [2] R. J. J. Mackenbach et al., J. Plasma Phys. 89, 905890513 (2023).
.. [3] K. Unalmis et al., "Spectrally accurate, reverse-mode differentiable
       bounce-averaging algorithm and its applications,"
       J. Plasma Physics. 2026;92(3):E72. https://doi.org/10.1017/S0022377826101652.

"""

from functools import partial

import numpy as np
from jax.lax import stop_gradient
from orthax import orthgauss
from orthax.recurrence import GeneralizedLaguerre
from quadax import quadgk

from desc.backend import jit, jnp

from ..integrals.bounce_integral import Bounce2D, Options
from ..utils import safediv, warnif
from ._drift import _binormal_drift, _radial_drift, _sqrt_G_hat
from .data_index import register_compute_fun


def _ae_1(G, G_ω_α, G_ω_ψ, data):
    """Compute energy-independent inputs for AE."""
    shape = (-1,) + (1,) * G.ndim

    G = G[..., None, :]  # This is sqrt G hat.
    # scale by conjugate widths
    G_ω_α = G_ω_α[..., None, :] * data["ae psi width"].reshape(shape)
    G_ω_ψ = G_ω_ψ[..., None, :] * data["ae alpha width"].reshape(shape)
    η_n = data["ae grad(density)"].reshape(shape)
    η_T = data["ae grad(temperature)"].reshape(shape)
    C = η_n - 1.5 * η_T

    drift = jnp.hypot(G_ω_α, G_ω_ψ)
    return G, G_ω_α, G_ω_ψ, η_T, C, drift


def _ae_2(G, G_ω_α, G_ω_ψ, η_T, C, drift, energy):
    """Evaluate the local AE kernel at fixed energy."""
    energy = energy[..., None]
    drive = jnp.hypot(G * (η_T + safediv(C, energy)) - G_ω_α, G_ω_ψ)
    return G_ω_α * C + (G_ω_α * η_T + safediv(drift * (drive - drift), G)) * energy


def _ae_E(G, G_ω_α, G_ω_ψ, η_T, C, drift, pitch_weight, energy):
    """Reduce the local AE kernel over wells, field lines, and pitch angles."""
    return (
        pitch_weight[..., None]
        * _ae_2(G, G_ω_α, G_ω_ψ, η_T, C, drift, energy).sum(-1).mean(-3)
    ).sum(-2)


def _energy_quad(num_energy):
    # The energy integral has weight E^(5/2) exp(-E), but
    # ω_* = η_T + C / E makes AE(E) ~ C/E for E near zero.
    return stop_gradient(orthgauss(num_energy, GeneralizedLaguerre(np.array([1.5]))))


@register_compute_fun(
    name="available energy",
    label="\\widehat{A}",
    units="~",
    units_long="None",
    description="Dimensionless available energy of trapped electrons",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "psi_r",
        "rho",
        "ne",
        "ne_r",
        "Te",
        "Te_r",
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
        "V_psi",
    ]
    + Bounce2D.required_names,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    radial_scale="float : Multiplier for the radial correlation length.",
    binormal_scale="float : Multiplier for the binormal correlation length.",
    quad_abs_err=(
        "float : Absolute tolerance for adaptive energy quadrature. "
        "If False, then this is interpreted as a flag to use a fixed quadrature, "
        "which is faster, but less accurate."
    ),
    quad_rel_err="float : Relative tolerance for adaptive energy quadrature.",
    energy_quad="tuple : Optional nodes and weights for fixed energy quadrature.",
    **Options._doc,
)
@partial(
    jit,
    static_argnames=Options._static_argnames + ("quad_abs_err", "quad_rel_err"),
)
def _available_energy(params, transforms, profiles, data, **kwargs):
    """Dimensionless available energy of trapped electrons [2]_.

    Parameters
    ----------
    radial_scale, binormal_scale : float
        Correlation-length multipliers. Default is 1.0.
    quad_abs_err, quad_rel_err : float or bool
        Tolerances for the adaptive energy quadrature. If ``quad_abs_err`` is
        False, then this is interpreted as a flag to use a fixed quadrature,
        which is faster, but less accurate.
        Default is 1e-6.

    """
    # noqa: unused dependency
    warnif(
        kwargs.get("pitch_batch_size", None) is not None,
        msg="pitch_batch_size is currently ignored by available energy.",
    )

    radial_scale = kwargs.get("radial_scale", 1.0)
    binormal_scale = kwargs.get("binormal_scale", 1.0)
    abs_err = kwargs.get("quad_abs_err", 1e-6)
    rel_err = kwargs.get("quad_rel_err", 1e-6)
    energy_quad = kwargs.get("energy_quad", None)
    if not abs_err and energy_quad is None:
        energy_quad = _energy_quad(32)

    grid = transforms["grid"]
    opts = Options.guess(-1, grid, **kwargs)

    def foreach_surface(data):
        pitch_inv, weight = Bounce2D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        weight /= pitch_inv**2
        G, G_ω_α, G_ω_ψ = Bounce2D(grid, data, data["angle"], **opts).integrate(
            [_sqrt_G_hat, _binormal_drift, _radial_drift],
            pitch_inv,
            data,
            names,
            num_well=opts.num_well,
            loop=opts.loop,
        )

        G, G_ω_α, G_ω_ψ, η_T, C, drift = _ae_1(G, G_ω_α, G_ω_ψ, data)
        if energy_quad is not None:
            return _ae_E(G, G_ω_α, G_ω_ψ, η_T, C, drift, weight, energy_quad[0]).dot(
                energy_quad[1]
            )

        return quadgk(
            lambda energy: (energy**1.5 * jnp.exp(-energy))
            * _ae_E(G, G_ω_α, G_ω_ψ, η_T, C, drift, weight, energy),
            jnp.array([0.0, jnp.inf]),
            epsabs=abs_err,
            epsrel=rel_err,
        )[0].squeeze()

    names = (
        "cvdrift (periodic)",
        "gbdrift (periodic)",
        "gbdrift (secular)/phi",
        "|grad(psi)|*kappa_g",
    )
    out = Bounce2D.batch(
        foreach_surface,
        data,
        grid,
        angle=kwargs["angle"],
        names=names,
        flux_data={
            "ae grad(density)": safediv(radial_scale * data["ne_r"], data["ne"]),
            "ae psi width": radial_scale * data["psi_r"],
            "ae alpha width": safediv(binormal_scale, data["rho"]),
            "ae grad(temperature)": safediv(radial_scale * data["Te_r"], data["Te"]),
        },
        batch_size=opts.surf_batch_size,
        shard_input_data=opts.shard_input_data,
    )
    assert out.ndim == 1

    scalar = jnp.sqrt(jnp.pi) * grid.NFP / (3 * opts.num_field_periods)
    data["available energy"] = grid.expand(scalar * out) / data["V_psi"]
    return data
