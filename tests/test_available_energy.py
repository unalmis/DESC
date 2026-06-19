"""Tests for available-energy analytic limits."""

import numpy as np
import pytest
from quadax import quadgk
from scipy.special import erf

from desc.backend import jnp
from desc.compute._turbulence import _ae_kernel, _ae_precompute, _ae_reduce


def _paper_F(c1):
    """Weighting function F(c1) from Eq. (4.6) of Rodriguez & Mackenbach."""
    c1 = np.asarray(c1, dtype=float)
    return (
        2 * np.sqrt(c1) * (15 + 4 * c1) * np.exp(-c1)
        + 3 * np.sqrt(np.pi) * (2 * c1 - 5) * erf(np.sqrt(c1))
    ) / (8 * c1**2)


def _analytic_ae_kernel(energy, omega_alpha, omega_star):
    """Evaluate DESC's local AE kernel in the omnigenous density-gradient limit."""
    G = jnp.array([[[1.0]]])
    G_omega_alpha = jnp.array([[[omega_alpha]]])
    G_omega_psi = jnp.array([[[0.0]]])
    data = {
        "ae psi width": jnp.array([1.0]),
        "ae alpha width": jnp.array([1.0]),
        "ae grad(density)": jnp.array([omega_star]),
        "ae grad(temperature)": jnp.array([0.0]),
    }
    ae_data = _ae_precompute(G, G_omega_alpha, G_omega_psi, data)
    return np.asarray(_ae_kernel(*ae_data, energy)).ravel()


def _ae_inputs(c1, omega_alpha):
    """Return simple analytic-limit AE inputs."""
    G = jnp.array([[[1.0]]])
    G_omega_alpha = jnp.array([[[omega_alpha]]])
    G_omega_psi = jnp.array([[[0.0]]])
    data = {
        "ae psi width": jnp.array([1.0]),
        "ae alpha width": jnp.array([1.0]),
        "ae grad(density)": jnp.array([c1 * omega_alpha]),
        "ae grad(temperature)": jnp.array([0.0]),
    }
    return G, G_omega_alpha, G_omega_psi, data


def _adaptive_energy_integral(c1, omega_alpha, abs_err=1e-11, rel_err=1e-11):
    """Integrate the analytic-limit AE kernel with adaptive quadrature."""
    ae_data = _ae_precompute(*_ae_inputs(c1, omega_alpha))
    value = quadgk(
        lambda energy: (energy**1.5 * jnp.exp(-energy))
        * _ae_reduce(*ae_data, jnp.ones(1), energy).squeeze(-1),
        jnp.array([0.0, jnp.inf]),
        epsabs=abs_err,
        epsrel=rel_err,
    )[0]
    return np.asarray(value).squeeze()


@pytest.mark.unit
def test_available_energy_kernel_matches_ramp_form():
    """In the omnigenous, density-gradient limit, _ae reduces to a ramp."""
    omega_alpha = 0.7
    c1 = 2.0
    omega_star = c1 * omega_alpha
    energy = jnp.asarray([0.1, 1.0, 1.9, 2.0, 2.1, 5.0])

    actual = _analytic_ae_kernel(energy, omega_alpha, omega_star)
    expected = 2 * omega_alpha**2 * np.maximum(c1 - np.asarray(energy), 0.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.unit
def test_counter_rotating_particles_do_not_contribute_to_ae():
    """The Heaviside factor in Eq. (4.5) is attained in the kernel."""
    energy = jnp.asarray([0.1, 1.0, 10.0])
    np.testing.assert_allclose(
        _analytic_ae_kernel(energy, omega_alpha=-0.7, omega_star=1.4),
        0.0,
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.unit
def test_available_energy_quadgk_integral_matches_paper_weight_function():
    """Adaptive energy quadrature matches Eq. (4.6)."""
    omega_alpha = 0.7
    c1 = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(
        [_adaptive_energy_integral(c, omega_alpha) for c in c1],
        [2 * (c * omega_alpha) ** 2 * _paper_F(c) for c in c1],
        rtol=1e-7,
        atol=1e-9,
    )
