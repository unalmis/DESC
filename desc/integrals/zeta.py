"""Locally corrected trapezoidal quadrature for singular surface integrals.

This module implements the unified zeta correction of Wu and Martinsson for
on-surface Laplace layer potentials on a doubly periodic tensor-product grid.
The implementation is written in JAX so the correction weights remain
differentiable with respect to the local first fundamental form.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import equinox as eqx
import numpy as np
from interpax import Interpolator2D
from jax.scipy.special import gammaincc, gammasgn
from scipy.constants import mu_0
from scipy.special import binom
from scipy.special import gammaln as scipy_gammaln
from scipy.special import gammasgn as scipy_gammasgn

from desc.backend import fori_loop, gammaln, jax, jnp, scan
from desc.batching import batch_map
from desc.utils import apply, dot, rpz2xyz, rpz2xyz_vec, safediv, safenorm, xyz2rpz_vec


@dataclass(frozen=True)
class _ZetaTerm:
    """A term of the form ``coefficient / r**power``."""

    power: int
    offset: int
    coefficient: callable


@dataclass(frozen=True)
class _ZetaKernel:
    """Description of a kernel supported by :func:`zeta_integral`."""

    terms: tuple[_ZetaTerm, ...]
    keys: tuple[str, ...]
    eval_keys: tuple[str, ...] = ()
    ndim: int = 1


@dataclass(frozen=True)
class _CorrectionSpec:
    """Static metadata for one zeta correction level."""

    m: int
    l1: int
    l2: int


def _source_phi(source_data, source_phi):
    return source_data["phi"] if source_phi is None else source_phi


def _dx(eval_data, source_data, source_phi=None):
    eval_x = rpz2xyz(jnp.stack((eval_data["R"], eval_data["phi"], eval_data["Z"]), -1))
    source_phi = _source_phi(source_data, source_phi)
    source_x = rpz2xyz(jnp.stack((source_data["R"], source_phi, source_data["Z"]), -1))
    if source_x.ndim == 2:
        return eval_x[:, None, :] - source_x[None, :, :]
    return eval_x[:, None, :] - source_x


_dx.keys = ("R", "phi", "Z")


def _coef_1_over_r(_eval_data, source_data, _dx, _source_phi=None):
    return source_data["|e_theta x e_zeta|"]


def _coef_monopole(_eval_data, source_data, _dx, _source_phi=None):
    return -source_data["|e_theta x e_zeta|"] * source_data["B0*n"] / (4 * jnp.pi)


def _coef_nr_over_r3(_eval_data, source_data, dx, source_phi=None):
    nJ = rpz2xyz_vec(
        source_data["e_theta x e_zeta"],
        phi=_source_phi(source_data, source_phi),
    )
    return dot(nJ, dx)


def _coef_dipole(eval_data, source_data, dx, source_phi=None):
    return (
        _dipole_geometry(
            eval_data,
            source_data,
            dx,
            source_phi,
        )
        * source_data["Phi (periodic)"]
    )


def _coef_dipole_plus_half(eval_data, source_data, dx, source_phi=None):
    out = _dipole_geometry(eval_data, source_data, dx, source_phi)
    return out * (
        source_data["Phi (periodic)"] - eval_data["Phi(x) (periodic)"][:, None]
    )


def _dipole_geometry(_eval_data, source_data, dx, source_phi=None):
    nJ = rpz2xyz_vec(
        source_data["e_theta x e_zeta"],
        phi=_source_phi(source_data, source_phi),
    )
    return dot(nJ, dx) / (4 * jnp.pi)


def _coef_biot_savart(_eval_data, source_data, dx, source_phi=None):
    K = rpz2xyz_vec(source_data["K_vc"], phi=_source_phi(source_data, source_phi))
    aK = source_data["|e_theta x e_zeta|"][..., None] * K
    return mu_0 * jnp.cross(aK, dx, axis=-1) / (4 * jnp.pi)


def _coef_biot_savart_A(_eval_data, source_data, _dx, source_phi=None):
    K = rpz2xyz_vec(source_data["K_vc"], phi=_source_phi(source_data, source_phi))
    return mu_0 * source_data["|e_theta x e_zeta|"][..., None] * K / (4 * jnp.pi)


def _coef_BS_plus_grad_S(_eval_data, source_data, dx, source_phi=None):
    K = rpz2xyz_vec(
        source_data["K_vc (periodic)"],
        phi=_source_phi(source_data, source_phi),
    )
    a = source_data["|e_theta x e_zeta|"]
    return (
        jnp.cross(a[..., None] * K, dx, axis=-1)
        + dx * (source_data["B0*n"] * a)[..., None]
    ) / (4 * jnp.pi)


class ZetaPlan(eqx.Module):
    """Quadrature data for one grid/discretization and kernel order."""

    ORDERS = frozenset((3, 5, 7, 9))
    DEFAULT_METRIC_INTERPOLATION = 33
    KERNELS = {
        "1_over_r": _ZetaKernel(
            terms=(_ZetaTerm(1, 0, _coef_1_over_r),),
            keys=_dx.keys + ("|e_theta x e_zeta|",),
        ),
        "monopole": _ZetaKernel(
            terms=(_ZetaTerm(1, 0, _coef_monopole),),
            keys=_dx.keys + ("B0*n", "|e_theta x e_zeta|"),
        ),
        "nr_over_r3": _ZetaKernel(
            terms=(_ZetaTerm(3, 1, _coef_nr_over_r3),),
            keys=_dx.keys + ("e_theta x e_zeta",),
        ),
        "dipole": _ZetaKernel(
            terms=(_ZetaTerm(3, 1, _coef_dipole),),
            keys=_dx.keys + ("e_theta x e_zeta", "Phi (periodic)"),
        ),
        "dipole_plus_half": _ZetaKernel(
            terms=(_ZetaTerm(3, 1, _coef_dipole_plus_half),),
            keys=_dx.keys + ("e_theta x e_zeta", "Phi (periodic)"),
            eval_keys=("Phi(x) (periodic)",),
        ),
        "biot_savart": _ZetaKernel(
            terms=(_ZetaTerm(3, 0, _coef_biot_savart),),
            keys=_dx.keys + ("K_vc", "|e_theta x e_zeta|"),
            ndim=3,
        ),
        "biot_savart_A": _ZetaKernel(
            terms=(_ZetaTerm(1, 0, _coef_biot_savart_A),),
            keys=_dx.keys + ("K_vc", "|e_theta x e_zeta|"),
            ndim=3,
        ),
        "biot_savart_grad_S": _ZetaKernel(
            terms=(_ZetaTerm(3, 0, _coef_BS_plus_grad_S),),
            keys=_dx.keys + ("K_vc (periodic)", "B0*n", "|e_theta x e_zeta|"),
            ndim=3,
        ),
    }

    num_theta: int = eqx.field(static=True)
    num_zeta: int = eqx.field(static=True)
    NFP: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    ht: float = eqx.field(static=True)
    hz: float = eqx.field(static=True)
    node_idx: jnp.ndarray
    terms: tuple[_ZetaTerm, ...] = eqx.field(static=True)
    ndim: int = eqx.field(static=True)
    eval_keys: frozenset[str] = eqx.field(static=True)
    source_keys: frozenset[str] = eqx.field(static=True)
    stencil_source_keys: frozenset[str] = eqx.field(static=True)
    correction_specs: tuple[tuple[_ZetaTerm, _CorrectionSpec], ...] = eqx.field(
        static=True
    )
    correction_u: tuple[jnp.ndarray, ...]
    correction_v: tuple[jnp.ndarray, ...]
    correction_fac: jnp.ndarray
    correction_qpow: jnp.ndarray
    correction_moment_inv: tuple[jnp.ndarray, ...]
    stencil_source_idx: tuple[jnp.ndarray, ...]
    stencil_zeta_shift: tuple[jnp.ndarray, ...]
    chunk_size: int | None = eqx.field(static=True)
    cutoff: int = eqx.field(static=True)
    metric_interpolation: int = eqx.field(static=True)
    metric_interpolators: tuple[Interpolator2D, ...] | None

    def hz_over_ht(self):
        """Return the toroidal-to-poloidal grid spacing ratio."""
        return self.hz / self.ht

    def period(self):
        """Return one field-period length in toroidal angle."""
        return self.hz * self.num_zeta

    def weight(self):
        """Return one tensor-product trapezoid quadrature weight."""
        return self.ht * self.hz

    @staticmethod
    def resolve_kernel_name(kernel):
        """Return the registered kernel name."""
        if isinstance(kernel, str):
            return kernel
        for name, candidate in ZetaPlan.KERNELS.items():
            if candidate is kernel:
                return name
        raise ValueError(f"Unsupported zeta kernel {kernel}.")

    @staticmethod
    def term_order_count(order, power, offset):
        """Return the highest correction index for one kernel term."""
        return (order + power + 1) // 2 - 2 - offset

    def zeros(self, num_eval):
        """Return a zero output block with the plan's scalar/vector shape."""
        shape = (num_eval, 3) if self.ndim == 3 else (num_eval,)
        return jnp.zeros(shape)

    def period_loop(self, fun):
        """Accumulate ``fun`` over all toroidal field-period shifts."""

        def body(j, out):
            return out + fun(j * self.period())

        return fori_loop(0, self.NFP, body, self.zeros(self.num_nodes))

    @staticmethod
    def term_correction_specs(order, power, offset):
        """Precompute static stencil metadata for one kernel term."""
        M = ZetaPlan.term_order_count(order, power, offset)
        if M < 0:
            return ()

        return tuple(
            _term_correction_spec(order, power, offset, m) for m in range(2 * M + 1)
        )

    @staticmethod
    def kernel_term_specs(kernel_name, order):
        """Return static term/correction metadata for one kernel and order."""
        kernel = ZetaPlan.KERNELS[kernel_name]
        return tuple(
            (term, ZetaPlan.term_correction_specs(order, term.power, term.offset))
            for term in kernel.terms
        )

    @staticmethod
    def compute_max_stencil_offset(correction_specs):
        """Return the largest integer grid offset used by correction specs."""
        return max((spec.l2 for _, spec in correction_specs), default=0)

    @staticmethod
    def check_stencil_fits_grid(grid, max_stencil_offset):
        """Require enough nodes that the local correction does not wrap globally."""
        min_nodes = 2 * max_stencil_offset + 5
        full_toroidal_nodes = grid.num_zeta * grid.NFP
        assert grid.num_theta >= min_nodes and full_toroidal_nodes >= min_nodes

    @staticmethod
    def paper_epstein_cutoff(source_data, grid):
        """Return the paper's metric-dependent Epstein lattice cutoff."""
        E = source_data["g_tt"]
        F = source_data["g_tz"]
        G = source_data["g_zz"]

        hz_over_ht = grid.num_theta / (grid.num_zeta * grid.NFP)
        F = hz_over_ht * F
        G = hz_over_ht**2 * G
        det = jnp.sqrt(E * G - F**2)
        E = E / det
        F = F / det
        G = G / det
        disc = jnp.hypot(E - G, 2 * F)
        lam_min = jnp.nanmin((E + G - disc) / 2)

        epstein_tail_exponent = 33.0
        cutoff = int(jnp.floor(jnp.sqrt(epstein_tail_exponent / jnp.pi / lam_min)) + 3)
        return max(1, cutoff)

    @staticmethod
    def resolve_epstein_cutoff(epstein_cutoff, source_data=None, grid=None):
        """Resolve fixed or paper-adaptive Epstein cutoff to a static integer."""
        auto_epstein_cutoff = 0
        auto_fallback_cutoff = 12
        if epstein_cutoff is not None and epstein_cutoff > auto_epstein_cutoff:
            return epstein_cutoff
        if source_data is not None and grid is not None:
            return ZetaPlan.paper_epstein_cutoff(source_data, grid)
        return auto_fallback_cutoff

    @staticmethod
    def resolve_metric_interpolation_size(metric_interpolation):
        """Resolve the number of shape-grid samples used for interpolation."""
        if metric_interpolation is None or metric_interpolation == 0:
            return 0
        assert metric_interpolation >= 2
        return metric_interpolation

    @staticmethod
    def build_metric_interpolators(
        correction_specs,
        correction_moment_inv,
        cutoff,
        metric_interpolation,
    ):
        """Build cached metric-shape interpolators for a plan."""
        if metric_interpolation == 0:
            return None
        return _metric_interpolation_tables(
            correction_specs,
            correction_moment_inv,
            cutoff,
            metric_interpolation,
        )

    def with_metric_interpolation(
        self,
        metric_interpolation=DEFAULT_METRIC_INTERPOLATION,
    ):
        """Return a copy with cached metric-shape interpolators rebuilt."""
        metric_interpolation = self.resolve_metric_interpolation_size(
            metric_interpolation
        )
        return replace(
            self,
            metric_interpolation=metric_interpolation,
            metric_interpolators=self.build_metric_interpolators(
                self.correction_specs,
                self.correction_moment_inv,
                self.cutoff,
                metric_interpolation,
            ),
        )

    @staticmethod
    def from_grid(
        grid,
        kernel,
        order,
        *,
        chunk_size=None,
        epstein_cutoff=12,
        metric_interpolation=DEFAULT_METRIC_INTERPOLATION,
    ):
        """Build a correction plan for a fixed meshgrid and kernel/order."""
        assert grid.can_fft2
        assert grid.num_rho == 1
        assert order in ZetaPlan.ORDERS
        kernel_name = ZetaPlan.resolve_kernel_name(kernel)
        kernel = ZetaPlan.KERNELS[kernel_name]
        term_specs = ZetaPlan.kernel_term_specs(kernel_name, order)
        correction_specs = tuple(
            (term, spec) for term, specs in term_specs for spec in specs
        )
        correction_offsets = tuple(
            _stencil_offsets(spec.l1, spec.l2) for _, spec in correction_specs
        )
        max_stencil_offset = ZetaPlan.compute_max_stencil_offset(correction_specs)
        ZetaPlan.check_stencil_fits_grid(grid, max_stencil_offset)
        chunk_size = _auto_chunk_size(grid, chunk_size)
        cutoff = ZetaPlan.resolve_epstein_cutoff(epstein_cutoff)
        metric_interpolation = ZetaPlan.resolve_metric_interpolation_size(
            metric_interpolation
        )
        ht = 2 * np.pi / grid.num_theta
        hz = 2 * np.pi / grid.num_zeta / grid.NFP
        period = hz * grid.num_zeta
        node_idx = jnp.arange(grid.num_nodes)
        stencil_data = tuple(
            _stencil_gather_indices(node_idx, grid.num_theta, grid.num_zeta, di, dj)
            for di, dj in correction_offsets
        )
        correction_u = tuple(ht * jnp.asarray(di) for di, _ in correction_offsets)
        correction_v = tuple(ht * jnp.asarray(dj) for _, dj in correction_offsets)
        correction_fac = jnp.asarray(
            tuple(
                _scale_correction_fac(binom(-term.power / 2, spec.m), ht, term.power)
                for term, spec in correction_specs
            )
        )
        correction_qpow = jnp.asarray(
            tuple(spec.m + term.power / 2 for term, spec in correction_specs)
        )
        correction_moment_inv = tuple(
            _moment_matrix_inverse(spec.l1, spec.l2) for _, spec in correction_specs
        )
        return ZetaPlan(
            num_theta=grid.num_theta,
            num_zeta=grid.num_zeta,
            NFP=grid.NFP,
            num_nodes=grid.num_nodes,
            ht=ht,
            hz=hz,
            node_idx=node_idx,
            terms=kernel.terms,
            ndim=kernel.ndim,
            eval_keys=_GRID_KEYS | frozenset(kernel.eval_keys),
            source_keys=_GRID_KEYS | _METRIC_KEYS | frozenset(kernel.keys),
            stencil_source_keys=_GRID_KEYS | frozenset(kernel.keys),
            correction_specs=correction_specs,
            correction_u=correction_u,
            correction_v=correction_v,
            correction_fac=correction_fac,
            correction_qpow=correction_qpow,
            correction_moment_inv=correction_moment_inv,
            stencil_source_idx=tuple(ind for ind, _ in stencil_data),
            stencil_zeta_shift=tuple(wrap_z * period for _, wrap_z in stencil_data),
            chunk_size=chunk_size,
            cutoff=cutoff,
            metric_interpolation=metric_interpolation,
            metric_interpolators=ZetaPlan.build_metric_interpolators(
                correction_specs,
                correction_moment_inv,
                cutoff,
                metric_interpolation,
            ),
        )


_GRID_KEYS = frozenset(_dx.keys)
_METRIC_KEYS = frozenset(("g_tt", "g_tz", "g_zz"))


def _scale_correction_fac(fac, ht, power):
    """Scale a term coefficient by the local grid spacing."""
    exponent = 2 - power
    if exponent < 0:
        reciprocal_exponent = -exponent
        return fac / ht**reciprocal_exponent
    return fac * ht**exponent


def _reduced_layer_offsets(layer):
    """Return the independent half of one diamond layer."""
    il = np.arange(layer, -layer, -1, dtype=int)
    jl = np.concatenate(
        (np.arange(layer, dtype=int), np.arange(layer, 0, -1, dtype=int))
    )
    return tuple(il), tuple(jl)


def _mirrored_layer_offsets(layer):
    """Return symmetric integer offset pairs for one diamond layer."""
    il, jl = _reduced_layer_offsets(layer)
    return il + tuple(-i for i in il), jl + tuple(-j for j in jl)


def _metric_interpolation_interpolator(
    term,
    spec,
    moment_inv,
    rho_axis,
    log_gamma_axis,
    E_shape_table,
    F_shape_table,
    G_shape_table,
    cutoff,
):
    """Build a single metric-shape interpolator for a correction spec."""
    tau = _intrinsic_weights(
        spec.l1,
        spec.l2,
        moment_inv,
        spec.m + term.power / 2,
        E_shape_table,
        F_shape_table,
        G_shape_table,
        cutoff,
    )
    tau = _expand_intrinsic_weights(tau, spec.l1, spec.l2).T
    table = tau.reshape(rho_axis.size, log_gamma_axis.size, tau.shape[-1])
    return Interpolator2D(
        rho_axis,
        log_gamma_axis,
        table,
        method="cubic2",
        extrap=True,
    )


def _term_correction_spec(order, power, offset, m):
    """Build a single correction spec tuple."""
    l1 = (3 * m + 1) // 2 + offset
    l2 = ZetaPlan.term_order_count(order, power, offset) + m + offset
    return _CorrectionSpec(
        m=m,
        l1=l1,
        l2=l2,
    )


class EpsteinZeta:
    """Static Epstein zeta evaluator and derivative helper namespace."""

    @staticmethod
    def scaled_upper_gamma(a, x):
        """Return the scaled upper incomplete gamma for static real ``a``."""
        if a > 0:
            return gammaincc(a, x) * jnp.exp(gammaln(a)) / x**a
        if a == 0:
            return jax.scipy.special.exp1(x)

        k = int(np.floor(-a) + 1)
        ap = a + k
        g = gammaincc(ap, x) * jnp.exp(gammaln(ap)) / x**ap
        ex = jnp.exp(-x)

        def body(carry, _):
            ap, g = carry
            ap = ap - 1
            return (ap, (g * x - ex) / ap), None

        return scan(body, (jnp.asarray(ap), g), None, length=k)[0][1]

    @staticmethod
    def scaled_upper_gamma_batch(k, s1, s2, x):
        """Return vectorized pair of scaled upper gamma terms."""
        return jnp.stack(
            [
                EpsteinZeta.scaled_upper_gamma(s1 + kk, x)
                + EpsteinZeta.scaled_upper_gamma(s2 + kk, x)
                for kk in k
            ]
        )

    @staticmethod
    def gamma(s):
        """Return ``Gamma(s)`` for static real ``s``."""
        return scipy_gammasgn(s) * np.exp(scipy_gammaln(s))

    @staticmethod
    def lattice(cutoff):
        """Return value-independent lattice points for the Epstein Ewald sum."""
        ij = np.arange(-cutoff, cutoff + 1)
        ii, jj = map(np.ravel, np.meshgrid(ij, ij, indexing="ij"))
        valid = ((ii != 0) | (jj != 0)) & (np.hypot(ii, jj) <= cutoff)
        return jnp.asarray(ii[valid]), jnp.asarray(jj[valid])

    @staticmethod
    def evaluate(s, E, F, G, cutoff=12):
        """Evaluate the Epstein zeta function ``Z(s; E, F, G)``."""
        if not s:
            return -jnp.ones_like(E)
        det = jnp.sqrt(E * G - F**2)
        E = E / det
        F = F / det
        G = G / det

        i, j = EpsteinZeta.lattice(cutoff)
        Q = E[..., None] * i**2 + 2 * F[..., None] * i * j
        Q = Q + G[..., None] * j**2
        x = jnp.pi * Q

        s1 = s / 2
        s2 = 1 - s1
        S = (
            EpsteinZeta.scaled_upper_gamma(s1, x)
            + EpsteinZeta.scaled_upper_gamma(s2, x)
        ).sum(axis=-1)
        gamma_s1 = gammasgn(s1) * jnp.exp(gammaln(s1))
        S = (S - 1 / s1 - 1 / s2) * (jnp.pi / det) ** s1 / gamma_s1
        return S

    @staticmethod
    def partial(s, orders, cutoff=12):
        """Build a vmapped partial derivative function for ``Z(s; E, F, G)``."""

        def fun(E, F, G):
            return EpsteinZeta.evaluate(s, E, F, G, cutoff=cutoff)

        for argnum, count in enumerate(orders):
            for _ in range(count):
                fun = jax.grad(fun, argnums=argnum)
        return jax.vmap(fun)

    @staticmethod
    def deriv(s, degree, E, F, G, cutoff=12):
        """Return derivatives ordered by monomials ``u**(2d-l) v**l``."""
        E = E.ravel()
        F = F.ravel()
        G = G.ravel()
        if not s:
            if degree == 0:
                return -jnp.ones_like(E)[None, :]
            return jnp.zeros((2 * degree + 1, E.size))
        if degree == 0:
            return EpsteinZeta.evaluate(s, E, F, G, cutoff=cutoff)[None, :]

        EF = EpsteinZeta.pair_partials(s, degree, E, F, G, "EF", cutoff)
        FG = EpsteinZeta.pair_partials(s, degree, E, F, G, "FG", cutoff)
        vals = [
            *(0.5**ell * EF[ell] for ell in range(degree + 1)),
            *(
                0.5 ** (2 * degree - ell) * FG[ell - degree]
                for ell in range(degree + 1, 2 * degree + 1)
            ),
        ]
        return jnp.stack(vals)

    @staticmethod
    def directional_unmixing(degree, pair):
        """Return directions and inverse map for directional derivatives."""
        angles = jnp.arange(degree + 1) * jnp.pi / (2 * degree)
        a = jnp.cos(angles)
        b = jnp.sin(angles)
        k = np.arange(degree + 1)
        A = binom(degree, k)[None, :] * (a[:, None] ** (degree - k[None, :]))
        A = A * b[:, None] ** k[None, :]
        inv = jnp.linalg.inv(A)
        dirs = jnp.zeros((degree + 1, 3))
        if pair == "EF":
            dirs = dirs.at[:, 0].set(a)
            dirs = dirs.at[:, 1].set(b)
        elif pair == "FG":
            dirs = dirs.at[:, 1].set(a)
            dirs = dirs.at[:, 2].set(b)
        else:
            raise ValueError(f"Unknown Epstein derivative pair {pair}.")

        return dirs, inv

    @staticmethod
    def partial_bell_table(xs, degree):
        """Return all partial Bell polynomials up to ``degree``."""
        zero = jnp.zeros_like(xs[0])
        one = jnp.ones_like(xs[0])
        table = [[zero for _ in range(degree + 1)] for _ in range(degree + 1)]
        table[0][0] = one
        for n in range(1, degree + 1):
            for m in range(1, n + 1):
                table[n][m] = sum(
                    (
                        binom(n - 1, i - 1) * xs[i - 1] * table[n - i][m - 1]
                        for i in range(1, n - m + 2)
                    ),
                    zero,
                )
        return table

    @staticmethod
    def bell_sums(xs, degree, coeff):
        """Return ``sum_i coeff**i B_{k,i}`` for ``k=0..degree``."""
        table = EpsteinZeta.partial_bell_table(xs, degree)
        zero = jnp.zeros_like(xs[0])
        return [table[0][0]] + [
            sum((coeff**i * table[k][i] for i in range(1, k + 1)), zero)
            for k in range(1, degree + 1)
        ]

    @staticmethod
    def pair_partials(s, degree, E, F, G, pair, cutoff):
        """Return ``degree``th partials in the EF or FG plane."""
        dirs, inv = EpsteinZeta.directional_unmixing(degree, pair)
        directional = EpsteinZeta.directional_derivative(
            s,
            degree,
            E,
            F,
            G,
            dirs,
            cutoff,
        )
        return inv @ directional

    @staticmethod
    def determinant_derivatives(H, K, degree):
        """Return determinant-log directional derivatives used by recurrence."""
        if degree <= 1:
            return (H,)

        H1 = -2 * H**2 + K
        if degree == 2:
            return (H, H1)

        def body(carry, k):
            prev2, prev1 = carry
            next_ = -2 * k * H * prev1 - k * (k - 1) * K * prev2
            return (prev1, next_), next_

        rest = scan(body, (H, H1), jnp.arange(2, degree))[1]
        return (H, H1, *rest)

    @staticmethod
    def directional_derivative(s, degree, E, F, G, dirs, cutoff):
        """Evaluate directional Epstein derivatives using Appendix B recurrences."""
        s1 = s / 2
        s2 = 1 - s1
        det = jnp.sqrt(E * G - F**2)
        C = (jnp.pi / det) ** s1 / EpsteinZeta.gamma(s1)

        ii, jj = EpsteinZeta.lattice(cutoff)
        q = E[:, None] * ii**2 + 2 * F[:, None] * ii * jj + G[:, None] * jj**2
        x = jnp.pi * q / det[:, None]

        gamma_vals = EpsteinZeta.scaled_upper_gamma_batch(
            range(degree + 1),
            s1,
            s2,
            x,
        )

        L = dirs[:, 0, None]
        M = dirs[:, 1, None]
        N = dirs[:, 2, None]
        E0 = E[None, :]
        F0 = F[None, :]
        G0 = G[None, :]
        D0 = det[None, :] ** 2
        H = (G0 * L + E0 * N - 2 * F0 * M) / (2 * D0)
        K = (L * N - M**2) / D0
        H_derivs = EpsteinZeta.determinant_derivatives(H, K, degree)

        A_minus = EpsteinZeta.bell_sums(H_derivs, degree, -1.0)
        A_plus = EpsteinZeta.bell_sums(H_derivs, degree, s1)

        qtilde = x[None, :, :]
        R = (
            L[:, :, None] * ii[None, None, :] ** 2
            + 2 * M[:, :, None] * ii[None, None, :] * jj[None, None, :]
            + N[:, :, None] * jj[None, None, :] ** 2
        )
        Rtilde = jnp.pi * R / det[None, :, None]
        Q_derivs = [None] + [
            (A_minus[k][:, :, None] * qtilde + k * A_minus[k - 1][:, :, None] * Rtilde)
            for k in range(1, degree + 1)
        ]

        A_plus = jnp.stack(EpsteinZeta.bell_sums(H_derivs, degree, s1))
        Z0 = C * (-1 / s1 - 1 / s2 + gamma_vals[0].sum(axis=-1))
        if degree == 0:
            return Z0[None, :]

        Z0 = jnp.broadcast_to(Z0[None, :], (degree + 1, Z0.shape[0]))
        Q_bell = EpsteinZeta.partial_bell_table(Q_derivs[1:], degree)
        zero_like_Z0 = jnp.zeros_like(Z0)
        G_derivs = jnp.stack(
            [zero_like_Z0]
            + [
                sum(
                    (
                        (-1) ** i * gamma_vals[i][None, :, :] * Q_bell[k][i]
                        for i in range(1, k + 1)
                    ),
                    jnp.zeros_like(qtilde),
                ).sum(axis=-1)
                for k in range(1, degree + 1)
            ],
        )

        z_acc = jnp.zeros((degree + 1,) + Z0.shape)
        z_acc = z_acc.at[0].set(Z0)
        indices = jnp.arange(degree)
        rows = np.arange(degree + 1)
        binom_coeffs = jnp.asarray([[binom(r, c) for c in rows] for r in rows])

        def recurrence_step(carry, k):
            coeff = binom_coeffs[k, :degree]
            valid = indices < k
            k_minus_i = jnp.where(valid, k - indices, 0)
            rows = carry[:degree]
            z_use = jnp.where(
                indices[:, None, None] == 0,
                rows[0][None, :, :],
                rows,
            )
            A_rows = jnp.where(
                valid[:, None, None],
                A_plus[k_minus_i],
                zero_like_Z0[None, :, :],
            )
            corr = jnp.einsum("k,k...", coeff, z_use * A_rows)
            term = C[None, :] * G_derivs[k]
            next_Z = term - corr
            return carry.at[k].set(next_Z), None

        z_acc, _ = scan(recurrence_step, z_acc, jnp.arange(1, degree + 1))
        return z_acc[degree]


def _stencil_offsets(l1, l2):
    """Return Wu-Martinsson diamond stencil offsets."""
    if l2 < l1:
        return (), ()

    start = 1 if l1 == 0 else l1
    i = [0] if l1 == 0 else []
    j = [0] if l1 == 0 else []
    for layer in range(start, l2 + 1):
        il, jl = _mirrored_layer_offsets(layer)
        i.extend(il)
        j.extend(jl)

    if l2 > 0:
        il = np.concatenate(
            (
                np.arange(l2, 0, -1, dtype=int),
                np.arange(-1, -l2 - 1, -1, dtype=int),
            )
        )
        jl = np.concatenate(
            (np.arange(1, l2 + 1, dtype=int), np.arange(l2, 0, -1, dtype=int))
        )
        i.extend(il)
        i.extend(-il)
        j.extend(jl)
        j.extend(-jl)
    return tuple(i), tuple(j)


def _moment_matrix_inverse(l1, l2):
    """Inverse moment-fitting map for a value-independent stencil."""
    power_range = range(2 * l1, 2 * l2 + 1, 2)
    powers_a = [q for p in power_range for q in range(p, -1, -1)]
    powers_b = [q for p in power_range for q in range(0, p + 1)]
    a = np.asarray(powers_a)[:, None]
    b = np.asarray(powers_b)[:, None]
    n = len(powers_a)
    A = np.zeros((n, n))

    u = np.arange(l1, -1, -1)[None, :]
    v = np.arange(0, l1 + 1)[None, :]
    block = u**a * v**b
    if l1 > 0:
        block = 2 * block
    if l1 > 1:
        block[:, 1:l1] = block[:, 1:l1] + (2 * ((-u[:, 1:l1]) ** a) * v[:, 1:l1] ** b)
    A[:, : l1 + 1] = block

    col = l1 + 1
    if l2 > l1:
        u = np.asarray(
            tuple(
                q
                for layer in range(l1 + 1, l2 + 1)
                for q in _reduced_layer_offsets(layer)[0]
            )
        )[None, :]
        v = np.asarray(
            tuple(
                q
                for layer in range(l1 + 1, l2 + 1)
                for q in _reduced_layer_offsets(layer)[1]
            )
        )[None, :]
        width = u.shape[1]
        A[:, col : col + width] = 2 * u**a * v**b
        col += width

    if l2 > 0:
        u = np.arange(l2, 0, -1)[None, :]
        v = np.arange(1, l2 + 1)[None, :]
        A[:, col:] = 2 * (u**a * v**b - (-u) ** a * v**b)

    return jnp.linalg.inv(A)


def _falling_factor(s, ell):
    s = jnp.asarray(s)

    if ell == 0:
        return jnp.ones_like(s)

    def body(out, k):
        return out * (-s - ell + 1 + k), None

    return scan(body, jnp.ones_like(s), jnp.arange(ell))[0]


def _intrinsic_weights(l1, l2, moment_inv, Qpow, E, F, G, cutoff):
    """Return reduced intrinsic zeta weights for one target or target batch."""
    W = jnp.concatenate(
        [
            -EpsteinZeta.deriv(2 * s, ell, E, F, G, cutoff) / _falling_factor(s, ell)
            for ell in range(l1, l2 + 1)
            for s in (Qpow - ell,)
        ]
    )
    return moment_inv @ W


def _expand_intrinsic_weights(tau, l1, l2):
    """Expand reduced weights to the full stencil returned by ``_stencil_offsets``."""
    inner = tau[: l1 + 1]
    if l1 > 1:
        inner = jnp.concatenate((inner, tau[l1 - 1 : 0 : -1]))
    if l1 > 0:
        inner = jnp.concatenate((inner, inner))

    middle_layers = tuple(range(l1 + 1, l2 + 1))
    starts = (
        np.cumsum((l1 + 1,) + tuple(2 * layer for layer in middle_layers[:-1]))
        if middle_layers
        else ()
    )
    middle = tuple(
        jnp.concatenate((seg, seg))
        for start, layer in zip(starts, middle_layers)
        for seg in (tau[start : start + 2 * layer],)
    )

    if l2 > 0:
        seg = tau[-l2:]
        outer = jnp.concatenate((seg, -seg[::-1]))
        outer = (jnp.concatenate((outer, outer)),)
    else:
        outer = ()
    pieces = (inner, *middle, *outer)
    return jnp.concatenate(pieces)


def _metric_shape_coords(E, F, G):
    """Return determinant scale and shape coordinates for a metric tensor."""
    det = jnp.sqrt(E * G - F**2)
    rho = F / jnp.sqrt(E * G)
    rho_limit = 1.0 - 1e-7
    rho = rho.clip(-rho_limit, rho_limit)
    log_gamma = jnp.log(G / E)
    return rho, log_gamma, det


def _log_gamma_to_table_coord(log_gamma):
    """Map unbounded ``log(G/E)`` to a bounded interpolation coordinate."""
    scale = 2.0
    return (2 / jnp.pi) * jnp.arctan(log_gamma / scale)


def _table_coord_to_log_gamma(coord):
    """Return the inverse of :func:`_log_gamma_to_table_coord`."""
    scale = 2.0
    return scale * jnp.tan(jnp.pi * coord / 2)


def _metric_interpolation_tables(correction_specs, correction_moment_inv, cutoff, size):
    """Tabulate intrinsic weights and build reusable metric-shape interpolators."""
    size = ZetaPlan.resolve_metric_interpolation_size(size)
    if size == 0:
        return None
    metric_log_gamma_coord_limit = 0.6
    rho_limit = 1.0 - 1e-7

    rho_axis = jnp.linspace(-rho_limit, rho_limit, size)
    log_gamma_axis = jnp.linspace(
        -metric_log_gamma_coord_limit,
        metric_log_gamma_coord_limit,
        size,
    )
    rho_grid, log_gamma_grid = map(
        jnp.ravel, jnp.meshgrid(rho_axis, log_gamma_axis, indexing="ij")
    )
    log_gamma_grid = _table_coord_to_log_gamma(log_gamma_grid)
    sqrt_gamma = jnp.exp(0.5 * log_gamma_grid)
    denom = jnp.sqrt(1 - rho_grid**2)
    E_shape_table = 1 / (sqrt_gamma * denom)
    F_shape_table = rho_grid / denom
    G_shape_table = sqrt_gamma / denom

    return tuple(
        _metric_interpolation_interpolator(
            term,
            spec,
            moment_inv,
            rho_axis,
            log_gamma_axis,
            E_shape_table,
            F_shape_table,
            G_shape_table,
            cutoff,
        )
        for (term, spec), moment_inv in zip(correction_specs, correction_moment_inv)
    )


def _intrinsic_weights_from_metric_table(E, F, G, interp, Qpow):
    """Interpolate expanded intrinsic weights and restore metric scale."""
    rho, log_gamma, scale = _metric_shape_coords(E, F, G)
    log_gamma = _log_gamma_to_table_coord(log_gamma)
    rho = rho.clip(interp.x[0], interp.x[-1])
    tau = interp(rho, log_gamma)
    return tau / scale[:, None] ** Qpow


def _metric_interpolation_weights(E, F, G, plan):
    """Evaluate interpolated intrinsic correction weights for one geometry."""
    if not plan.metric_interpolation:
        return None
    assert plan.metric_interpolators is not None

    return tuple(
        _intrinsic_weights_from_metric_table(
            E,
            F,
            G,
            plan.metric_interpolators[flat_idx],
            plan.correction_qpow[flat_idx],
        )
        for flat_idx in range(len(plan.correction_specs))
    )


def _metric_interpolation_weights_for_source(source_data, plan):
    """Precompute interpolated intrinsic weights for concrete source metrics."""
    if plan.metric_interpolation == 0:
        return None
    hz_over_ht = plan.hz_over_ht()
    return _metric_interpolation_weights(
        source_data["g_tt"],
        hz_over_ht * source_data["g_tz"],
        hz_over_ht**2 * source_data["g_zz"],
        plan,
    )


def _apply_term(term, eval_data, source_data, dx, weight, source_phi=None):
    coeff = term.coefficient(eval_data, source_data, dx, source_phi)
    denom = safenorm(dx, axis=-1) ** term.power
    if coeff.ndim > denom.ndim:
        denom = denom[..., None]
    return weight * safediv(coeff, denom)


def _is_dipole_density_term(term):
    return term.coefficient in (_coef_dipole, _coef_dipole_plus_half)


def _density_sum(weights, density):
    """Contract quadrature weights against scalar source density values."""
    if density.ndim == weights.ndim - 1:
        return weights.dot(density)
    if density.ndim == weights.ndim:
        return (weights * density).sum(axis=-1)
    raise ValueError("Unsupported scalar density shape for zeta quadrature.")


def _density_jump_sum(weights, eval_density, source_density):
    out = _density_sum(weights, source_density)
    return out - weights.sum(1) * eval_density


def _apply_density_term(term, eval_data, source_data, dx, weight, source_phi=None):
    weights = _dipole_geometry(eval_data, source_data, dx, source_phi)
    denom = safenorm(dx, axis=-1) ** term.power
    weights = weight * safediv(weights, denom)
    is_dipole = term.coefficient is _coef_dipole
    if is_dipole:
        return _density_sum(weights, source_data["Phi (periodic)"])
    return _density_jump_sum(
        weights,
        eval_data["Phi(x) (periodic)"],
        source_data["Phi (periodic)"],
    )


def _apply_density_correction(term, eval_data, source_data, dx, corr, source_phi=None):
    weights = corr * _dipole_geometry(eval_data, source_data, dx, source_phi)
    if term.coefficient is _coef_dipole:
        return _density_sum(weights, source_data["Phi (periodic)"])
    return _density_jump_sum(
        weights,
        eval_data["Phi(x) (periodic)"],
        source_data["Phi (periodic)"],
    )


def _prepare_zeta_data(eval_data, source_data, plan):
    eval_data, source_data = (
        apply(eval_data, jnp.asarray, subset=plan.eval_keys),
        apply(source_data, jnp.asarray, subset=plan.source_keys),
    )
    assert eval_data["R"].size == plan.num_nodes
    assert source_data["R"].size == plan.num_nodes
    return eval_data, source_data


def _auto_chunk_size(grid, chunk_size):
    if chunk_size == 0:
        return None
    return min(grid.num_nodes, 64) if chunk_size is None else chunk_size


def _stencil_gather_indices(idx, num_theta, num_zeta, di, dj):
    """Compute source indices and zeta wraps for one stencil spec."""
    di = jnp.asarray(di)
    dj = jnp.asarray(dj)
    zeta_idx, theta_idx = jnp.divmod(idx, num_theta)
    zeta_unwrapped = zeta_idx[:, None] + dj[None, :]
    wrap_z, mod_z = jnp.divmod(zeta_unwrapped, num_zeta)
    mod_t = (theta_idx[:, None] + di[None, :]) % num_theta
    ind = mod_t + num_theta * mod_z
    return ind, wrap_z


def _punctured_part(eval_data, source_data, plan):
    weight = plan.weight()

    def period_matvec(shift):
        source_phi_k = source_data["phi"] + shift

        def eval_chunk(eval_chunk_data):
            dx = _dx(eval_chunk_data, source_data, source_phi_k)

            def term_eval(term):
                if _is_dipole_density_term(term):
                    return _apply_density_term(
                        term,
                        eval_chunk_data,
                        source_data,
                        dx,
                        weight,
                        source_phi_k,
                    )
                term_val = _apply_term(
                    term,
                    eval_chunk_data,
                    source_data,
                    dx,
                    weight,
                    source_phi_k,
                )
                return term_val.sum(axis=1)

            return sum(term_eval(term) for term in plan.terms)

        return batch_map(eval_chunk, eval_data, plan.chunk_size)

    return plan.period_loop(period_matvec)


def _precomputed_correction_part(eval_data, source_data, plan, correction_weights):
    """Apply precomputed local DLP correction weights to the current density."""
    eval_density = eval_data["Phi(x) (periodic)"]
    source_density = source_data["Phi (periodic)"]

    def correction_for_spec(flat_idx, term, _spec):
        weights = correction_weights[flat_idx]
        density = source_density[plan.stencil_source_idx[flat_idx]]
        if term.coefficient is _coef_dipole:
            return (weights * density).sum(axis=1)
        return (weights * (density - eval_density[:, None])).sum(axis=1)

    return sum(
        correction_for_spec(flat_idx, term, spec)
        for flat_idx, (term, spec) in enumerate(plan.correction_specs)
    )


def _dlp_spec_data(
    chunk,
    idx,
    E,
    F,
    G,
    spec,
    flat_idx,
    plan,
    metric_weights,
    stencil_source_data,
    compute_geometry=False,
):
    """Compute per-spec local correction data for dipole Laplace DLP terms."""
    u = plan.correction_u[flat_idx]
    v = plan.correction_v[flat_idx]
    fac = plan.correction_fac[flat_idx]

    ind = plan.stencil_source_idx[flat_idx][idx]
    zeta_shift = plan.stencil_zeta_shift[flat_idx][idx]
    source_sten = {key: val[ind] for key, val in stencil_source_data.items()}
    source_phi_sten = source_sten["phi"] + zeta_shift

    dx = _dx(chunk, source_sten, source_phi_sten)
    if metric_weights is None:
        tau = _intrinsic_weights(
            spec.l1,
            spec.l2,
            plan.correction_moment_inv[flat_idx],
            np.asarray(plan.correction_qpow)[flat_idx].item(),
            E,
            F,
            G,
            plan.cutoff,
        )
        tau = _expand_intrinsic_weights(tau, spec.l1, spec.l2).T
    else:
        tau = metric_weights[flat_idx][idx]

    Q = (
        E[:, None] * u[None, :] ** 2
        + 2 * F[:, None] * u[None, :] * v[None, :]
        + G[:, None] * v[None, :] ** 2
    )
    r2mQh2 = (dot(dx, dx) - Q) / plan.ht**2
    corr = fac * tau * r2mQh2**spec.m
    geom = (
        _dipole_geometry(chunk, source_sten, dx, source_phi_sten)
        if compute_geometry
        else None
    )
    return corr, source_sten, source_phi_sten, dx, geom


def _correction_part(eval_data, source_data, plan, metric_weights=None):
    hz_over_ht = plan.hz_over_ht()
    E_all = source_data["g_tt"]
    F_all = hz_over_ht * source_data["g_tz"]
    G_all = hz_over_ht**2 * source_data["g_zz"]
    stencil_source_data = apply(source_data, subset=plan.stencil_source_keys)

    def eval_chunk(chunk):
        idx = chunk.pop("__idx")
        E = E_all[idx]
        F = F_all[idx]
        G = G_all[idx]

        def correction_for_spec(flat_idx, term, spec):
            corr, source_sten, source_phi_sten, dx, _ = _dlp_spec_data(
                chunk,
                idx,
                E,
                F,
                G,
                spec,
                flat_idx,
                plan,
                metric_weights,
                stencil_source_data,
            )
            if _is_dipole_density_term(term):
                return _apply_density_correction(
                    term,
                    chunk,
                    source_sten,
                    dx,
                    hz_over_ht * corr,
                    source_phi_sten,
                )
            coeff = (
                term.coefficient(
                    chunk,
                    source_sten,
                    dx,
                    source_phi_sten,
                )
                * hz_over_ht
            )
            return jnp.einsum("bs,bs...->b...", corr, coeff)

        return sum(
            correction_for_spec(flat_idx, term, spec)
            for flat_idx, (term, spec) in enumerate(plan.correction_specs)
        )

    chunk_input = eval_data.copy()
    chunk_input["__idx"] = plan.node_idx
    return batch_map(eval_chunk, chunk_input, plan.chunk_size)


def zeta_integral(eval_data, source_data, plan):
    """Evaluate a zeta-corrected on-surface integral from a prebuilt plan.

    Parameters
    ----------
    eval_data, source_data : dict
        Full-grid data at evaluation and source nodes.
    plan : ZetaPlan
        Precomputed local-correction and grid-layout information for the target
        discretization.

    Returns
    -------
    ndarray
        Corrected integral values on the full plan grid.

    """
    eval_data, source_data = _prepare_zeta_data(eval_data, source_data, plan)

    metric_weights = _metric_interpolation_weights_for_source(
        source_data,
        plan,
    )
    out = _punctured_part(eval_data, source_data, plan)
    out = out + _correction_part(
        eval_data,
        source_data,
        plan,
        metric_weights,
    )
    if plan.ndim == 3:
        out = xyz2rpz_vec(out, phi=eval_data["phi"])
    return out


def zeta_correction_weights(eval_data, source_data, plan):
    """Precompute geometry-dependent local DLP correction weights."""
    assert source_data["R"].size == plan.num_nodes
    assert eval_data["R"].size == plan.num_nodes
    eval_data = apply(eval_data, jnp.asarray, subset=_GRID_KEYS)
    source_data = apply(
        source_data,
        jnp.asarray,
        subset=_GRID_KEYS | _METRIC_KEYS | frozenset(("e_theta x e_zeta",)),
    )
    metric_weights = _metric_interpolation_weights_for_source(
        source_data,
        plan,
    )
    hz_over_ht = plan.hz_over_ht()
    E_all = source_data["g_tt"]
    F_all = hz_over_ht * source_data["g_tz"]
    G_all = hz_over_ht**2 * source_data["g_zz"]
    stencil_source_data = apply(
        source_data,
        subset=_GRID_KEYS | frozenset(("e_theta x e_zeta",)),
    )

    def eval_chunk(chunk):
        idx = chunk.pop("__idx")
        E = E_all[idx]
        F = F_all[idx]
        G = G_all[idx]

        def weights_for_spec(flat_idx, _term, spec):
            corr, _, _, dx, geom = _dlp_spec_data(
                chunk,
                idx,
                E,
                F,
                G,
                spec,
                flat_idx,
                plan,
                metric_weights,
                stencil_source_data,
                compute_geometry=True,
            )
            return hz_over_ht * corr * geom

        return tuple(
            weights_for_spec(flat_idx, _term, spec)
            for flat_idx, (_term, spec) in enumerate(plan.correction_specs)
        )

    chunk_input = eval_data.copy()
    chunk_input["__idx"] = plan.node_idx
    return batch_map(eval_chunk, chunk_input, plan.chunk_size)


def zeta_apply_correction_weights(eval_data, source_data, plan, correction_weights):
    """Apply ``D[Phi] + Phi/2`` using precomputed local correction weights."""
    eval_data, source_data = _prepare_zeta_data(eval_data, source_data, plan)
    return _punctured_part(eval_data, source_data, plan) + _precomputed_correction_part(
        eval_data,
        source_data,
        plan,
        correction_weights,
    )
