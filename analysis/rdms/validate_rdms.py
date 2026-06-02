"""
Validate built RDMs in analysis/results/rdms/.

Checks every RDM that is present; skips ones not yet built.
Exits 0 if all present RDMs pass; exits 1 if any check fails.

Run from repo root (with .venv active):
    python -m analysis.rdms.validate_rdms

To validate a specific subset:
    python -m analysis.rdms.validate_rdms --only sens_pre sem_km
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from analysis.rdms.common import RESULTS_DIR, _EXPECTED_LEN, _EXPECTED_N, load_rdm

_ALL_NAMES = ["sens_pre", "sens_post", "sem_km", "sem_wn", "clip_pre", "clip_post"]
_N = _EXPECTED_N       # 725
_WN_FALLBACK = 30.0    # must match semantic_wn._WN_FALLBACK_DIST
_KM_MAX_DEPTH = 8      # generous upper bound for this dataset's hierarchy


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class CheckFailed(AssertionError):
    """Raised when a named invariant is violated."""


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise CheckFailed(msg)


# ---------------------------------------------------------------------------
# Universal checks (every RDM)
# ---------------------------------------------------------------------------

def check_universal(name: str, d: np.ndarray) -> None:
    """Invariants that must hold for any correctly-built condensed RDM."""
    _check(d.ndim == 1, f"expected 1-D array, got shape {d.shape}")
    _check(len(d) == _EXPECTED_LEN, f"expected length {_EXPECTED_LEN}, got {len(d)}")
    n_nan = int(np.sum(np.isnan(d)))
    n_inf = int(np.sum(np.isinf(d)))
    _check(n_nan == 0, f"{n_nan} NaN values")
    _check(n_inf == 0, f"{n_inf} Inf values")
    _check(bool((d >= 0).all()), f"{int(np.sum(d < 0))} negative values")
    sq = squareform(d)
    _check(sq.shape == (_N, _N), f"squareform shape {sq.shape} != ({_N}, {_N})")
    _check(np.allclose(sq, sq.T), "distance matrix is not symmetric")
    _check(bool(np.all(np.diag(sq) == 0)), "diagonal has non-zero entries")


# ---------------------------------------------------------------------------
# Per-RDM checks
# ---------------------------------------------------------------------------

def check_sens(d: np.ndarray) -> None:
    """Pixel Euclidean distances should be strictly positive (non-identical images)."""
    _check(float(d.max()) > 0, "all pixel distances are zero — images may not have loaded")


def check_sens_correlation(d_pre: np.ndarray, d_post: np.ndarray) -> None:
    """Pre- and post-SHINE sensory RDMs should be strongly correlated (same objects)."""
    rho = float(spearmanr(d_pre, d_post).statistic)
    _check(rho > 0.9, f"Spearman rho(sens_pre, sens_post) = {rho:.3f} < 0.9")


def check_sem_km(d: np.ndarray) -> None:
    """KM distances are positive integers; max bounded by 2 * hierarchy depth."""
    _check(bool(np.allclose(d, np.round(d))), "KM distances are not all integers")
    _check(float(d.min()) >= 1.0, f"minimum off-diagonal KM distance {float(d.min()):.1f} < 1")
    max_d = float(d.max())
    _check(max_d <= 2 * _KM_MAX_DEPTH,
           f"max KM distance {max_d:.0f} > {2 * _KM_MAX_DEPTH} (2 × max_depth)")


def check_sem_wn(d: np.ndarray) -> None:
    """WN distances in [0, fallback]; fallback-pair fraction < 10%."""
    max_d = float(d.max())
    _check(max_d <= _WN_FALLBACK + 1e-6,
           f"max WN distance {max_d:.1f} > fallback {_WN_FALLBACK}")
    fallback_frac = float(np.mean(np.isclose(d, _WN_FALLBACK)))
    _check(fallback_frac < 0.10,
           f"{fallback_frac:.1%} of pairs use fallback distance (threshold: 10%)")


def check_clip(d: np.ndarray) -> None:
    """CLIP cosine distances must lie in [0, 2]."""
    max_d = float(d.max())
    _check(max_d <= 2.0 + 1e-6, f"max CLIP cosine distance {max_d:.4f} > 2.0")


def check_clip_correlation(d_pre: np.ndarray, d_post: np.ndarray) -> None:
    """Pre- and post-SHINE CLIP RDMs should be strongly correlated (same objects)."""
    rho = float(spearmanr(d_pre, d_post).statistic)
    _check(rho > 0.9, f"Spearman rho(clip_pre, clip_post) = {rho:.3f} < 0.9")


# ---------------------------------------------------------------------------
# Loader (skips absent RDMs; surfaces load errors as check failures)
# ---------------------------------------------------------------------------

def _try_load(name: str) -> tuple[np.ndarray | None, str | None]:
    """
    Returns (array, None) on success.
    Returns (None, None) if the .npy file is absent (not yet built — skip).
    Returns (None, error_msg) if the file exists but load_rdm() raises.
    """
    path = RESULTS_DIR / f"D_{name}.npy"
    if not path.exists():
        return None, None
    try:
        return load_rdm(name), None
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks(names: list[str] | None = None) -> bool:
    """
    Run all checks for the given names (default: all known RDMs).

    Returns True if every present RDM passes; False if any check fails.
    """
    targets = names if names is not None else _ALL_NAMES

    # Load phase
    present: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    load_errors: list[tuple[str, str]] = []

    for name in targets:
        d, err = _try_load(name)
        if err is not None:
            load_errors.append((name, err))
        elif d is None:
            skipped.append(name)
        else:
            present[name] = d

    if skipped:
        print(f"Skipped (not yet built): {', '.join(skipped)}")
    for name, err in load_errors:
        print(f"  LOAD ERROR  {name}: {err}")

    if not present and not load_errors:
        print("No RDMs found. Run `python -m analysis.rdms.build_all` first.")
        return False

    # Check phase
    failures: list[tuple[str, str]] = []

    def run(label: str, fn, *args) -> None:
        try:
            fn(*args)
            print(f"  PASS  {label}")
        except CheckFailed as exc:
            print(f"  FAIL  {label}: {exc}")
            failures.append((label, str(exc)))

    print("\n=== Universal checks ===")
    for name, d in present.items():
        run(f"{name}: shape / finite / non-negative / symmetric", check_universal, name, d)

    print("\n=== Per-RDM checks ===")
    for name, d in present.items():
        if name in ("sens_pre", "sens_post"):
            run(f"{name}: pixel distances > 0", check_sens, d)
        elif name == "sem_km":
            run(f"{name}: integer distances, min >= 1, max bounded", check_sem_km, d)
        elif name == "sem_wn":
            run(f"{name}: range [0, {_WN_FALLBACK}], fallback fraction < 10%", check_sem_wn, d)
        elif name in ("clip_pre", "clip_post"):
            run(f"{name}: cosine distances in [0, 2]", check_clip, d)

    print("\n=== Cross-variant checks ===")
    if "sens_pre" in present and "sens_post" in present:
        run("sens_pre vs sens_post: Spearman rho > 0.9",
            check_sens_correlation, present["sens_pre"], present["sens_post"])
    elif {"sens_pre", "sens_post"} & set(targets):
        print("  SKIP  sens correlation — need both sens_pre and sens_post")

    if "clip_pre" in present and "clip_post" in present:
        run("clip_pre vs clip_post: Spearman rho > 0.9",
            check_clip_correlation, present["clip_pre"], present["clip_post"])
    elif {"clip_pre", "clip_post"} & set(targets):
        print("  SKIP  clip correlation — need both clip_pre and clip_post")

    # Summary
    n_checked = len(present)
    n_fail = len(failures) + len(load_errors)
    print(f"\n{'=' * 50}")
    if n_fail:
        print(f"FAILED: {n_fail} issue(s) across {n_checked} RDM(s)")
        return False
    print(f"All checks passed ({n_checked} RDM(s) validated)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate built RDMs in analysis/results/rdms/.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", dest="names",
        help="Validate only these RDMs (default: all).",
    )
    args = parser.parse_args()
    if args.names:
        unknown = set(args.names) - set(_ALL_NAMES)
        if unknown:
            parser.error(f"Unknown RDM name(s): {sorted(unknown)}. Choose from: {_ALL_NAMES}")
    ok = run_all_checks(args.names)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
