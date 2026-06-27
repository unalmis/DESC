"""Benchmark/profile the FreeSurfaceError objective."""

import argparse
import csv
import time

import numpy as np

from desc.backend import jax
from desc.compute._laplace import Options as LaplaceOptions
from desc.examples import get
from desc.grid import LinearGrid
from desc.magnetic_fields import FreeSurfaceOuterField, ToroidalMagneticField
from desc.objectives import ForceBalance, FreeSurfaceError, ObjectiveFunction
from desc.optimize import ProximalProjection

_QUADRATURE_LABEL = {"singular": "malhotra", "zeta": "zeta"}


def _block_until_ready(out):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out,
    )


def _time_call(fun, repeats, warmup=False):
    if warmup:
        _block_until_ready(fun())
    times = []
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = _block_until_ready(fun())
        times.append(time.perf_counter() - start)
    return times, out


def _laplace_options(args, quadrature):
    options = {
        "solve_method": args.solve_method,
        "quadrature": quadrature,
        "zeta_order": args.zeta_order,
        "zeta_epstein_cutoff": args.zeta_epstein_cutoff,
        "zeta_metric_interpolation": args.zeta_metric_interpolation,
        "max_steps": args.max_steps,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    return LaplaceOptions(**options)


def _build_problem(args, quadrature):
    eq = get("W7-X")
    grid = LinearGrid(
        rho=np.array([1.0]),
        M=args.grid_m,
        N=args.grid_n,
        NFP=eq.NFP,
        sym=False,
    )
    B_coil = ToroidalMagneticField(5, 1)
    field = FreeSurfaceOuterField(
        eq.surface,
        M=args.field_m,
        N=args.field_n,
        B_coil=B_coil,
    )
    obj = ObjectiveFunction(
        [
            FreeSurfaceError(
                eq,
                field,
                grid=grid,
                options=_laplace_options(args, quadrature),
                deriv_mode=args.deriv_mode,
            )
        ]
    )
    constraint = ObjectiveFunction([ForceBalance(eq)])
    prox = ProximalProjection(
        obj,
        constraint,
        eq,
        solve_options={"solve_during_proximal_build": False},
    )
    start = time.perf_counter()
    prox.build()
    build_time = time.perf_counter() - start
    return prox, prox.x(eq), build_time


def _summarize_times(times):
    return {
        "min_s": min(times),
        "mean_s": sum(times) / len(times),
        "max_s": max(times),
        "runs": len(times),
    }


def _benchmark_quadrature(args, quadrature):
    prox, x, build_time = _build_problem(args, quadrature)
    compute_times, err = _time_call(
        lambda: prox.compute_scaled_error(x),
        args.compute_repeats,
        warmup=args.warmup_compute,
    )
    jac_error = ""
    if args.jac_repeats > 0:
        try:
            jac_times, _ = _time_call(
                lambda: prox.jac_scaled_error(x),
                args.jac_repeats,
                warmup=args.warmup_jac,
            )
            jac = _summarize_times(jac_times)
        except BaseException as err_jac:
            if args.fail_fast or isinstance(err_jac, (KeyboardInterrupt, SystemExit)):
                raise
            jac = {"min_s": np.nan, "mean_s": np.nan, "max_s": np.nan, "runs": 0}
            jac_error = str(err_jac).splitlines()[0]
    else:
        jac = {"min_s": np.nan, "mean_s": np.nan, "max_s": np.nan, "runs": 0}
        jac_error = "skipped"
    compute = _summarize_times(compute_times)
    return {
        "quadrature": quadrature,
        "label": _QUADRATURE_LABEL[quadrature],
        "grid_m": args.grid_m,
        "grid_n": args.grid_n,
        "zeta_order": args.zeta_order if quadrature == "zeta" else "",
        "zeta_metric_interpolation": (
            args.zeta_metric_interpolation if quadrature == "zeta" else ""
        ),
        "build_s": build_time,
        "compute_min_s": compute["min_s"],
        "compute_mean_s": compute["mean_s"],
        "compute_max_s": compute["max_s"],
        "compute_runs": compute["runs"],
        "jac_min_s": jac["min_s"],
        "jac_mean_s": jac["mean_s"],
        "jac_max_s": jac["max_s"],
        "jac_runs": jac["runs"],
        "error_size": int(err.size),
        "jac_error": jac_error,
    }


def _write_csv(path, rows):
    if path is None:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _print_results(rows):
    header = (
        "quadrature,label,grid_m,grid_n,zeta_order,zeta_metric_interpolation,"
        "build_s,compute_mean_s,compute_min_s,compute_runs,jac_mean_s,jac_runs,"
        "error_size,jac_error"
    )
    print(header)
    for row in rows:
        print(
            ",".join(
                (
                    row["quadrature"],
                    row["label"],
                    str(row["grid_m"]),
                    str(row["grid_n"]),
                    str(row["zeta_order"]),
                    str(row["zeta_metric_interpolation"]),
                    f"{row['build_s']:.6g}",
                    f"{row['compute_mean_s']:.6g}",
                    f"{row['compute_min_s']:.6g}",
                    str(row["compute_runs"]),
                    f"{row['jac_mean_s']:.6g}",
                    str(row["jac_runs"]),
                    str(row["error_size"]),
                    row["jac_error"].replace(",", ";"),
                )
            )
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FreeSurfaceError with Malhotra-style singular quadrature "
            "and zeta quadrature."
        )
    )
    parser.add_argument(
        "--quadratures",
        default="singular,zeta",
        help="Comma-separated subset of singular,zeta.",
    )
    parser.add_argument("--grid-m", type=int, default=8)
    parser.add_argument("--grid-n", type=int, default=8)
    parser.add_argument("--field-m", type=int, default=8)
    parser.add_argument("--field-n", type=int, default=8)
    parser.add_argument("--solve-method", default="gmres")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--zeta-order",
        type=int,
        default=3,
        help="Default 3 fits the original M=N=8 profile grid; use larger grids for 9.",
    )
    parser.add_argument("--zeta-epstein-cutoff", type=int, default=0)
    parser.add_argument(
        "--zeta-metric-interpolation",
        type=int,
        default=LaplaceOptions().zeta_metric_interpolation,
    )
    parser.add_argument("--deriv-mode", default="fwd")
    parser.add_argument("--compute-repeats", type=int, default=3)
    parser.add_argument("--jac-repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--warmup-jac",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    quadratures = [item.strip() for item in args.quadratures.split(",") if item.strip()]
    unknown = sorted(set(quadratures) - set(_QUADRATURE_LABEL))
    if unknown:
        raise ValueError(f"Unknown quadrature labels: {unknown}.")

    if args.profile_dir is None:
        rows = [_benchmark_quadrature(args, quadrature) for quadrature in quadratures]
    else:
        with jax.profiler.trace(args.profile_dir):
            with jax.profiler.TraceAnnotation("Benchmarking FreeSurfaceError"):
                rows = [
                    _benchmark_quadrature(args, quadrature)
                    for quadrature in quadratures
                ]
    _print_results(rows)
    _write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
