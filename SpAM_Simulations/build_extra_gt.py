"""Build an additional ground-truth embedding at a chosen dimensionality.

The stage-1 scan selects a dimensionality by out-of-sample agreement, and on the pilot it selects
**3**. That number is not the intrinsic dimensionality of the perceptual space: it is the
dimensionality at which *20 subjects at 17% pair coverage* still agree with another 20. Above D~4 the
fit has more freedom than the data constrains, the two halves diverge, and the curve falls - which is
overfitting, not evidence that the higher dimensions are empty. Two things corroborate that reading:
the top-5% closest-pair Jaccard climbs monotonically all the way to the largest candidate (D=20)
instead of turning over, and the peak global agreement is only rho=0.233, i.e. two halves share about
5% of rank variance, so power rather than geometry is the binding constraint.

**Why that matters for a planning simulation.** A 3-D ground truth is easier to recover than the
truth, so simulated subjects generated from it make required-N look smaller and closest-pair recovery
look better than they will be. For a study-planning simulation, erring optimistic is the wrong
direction, so stage 2 defaults to a *higher*-dimensional GT than the scan selected.

This script does not touch ``gt/selection.json``. That file records what the evidence chose and
should stay that way; the deliberate departure from it is stage 2's ``GT_NDIM``, which is documented
where it is used. What is written here is an extra ``gt_pre_shine_d{K}.npy`` plus an
``gt/extra_gts.json`` note recording that it was built, by whom, and why.

Run it inside an existing stage-1 clone, where R and the pilot data are already present::

    cd ~/spam_run/repo
    python -m SpAM_Simulations.build_extra_gt --ndim 8
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


def build(ndim: int, pilot_dir: str = "data", manifest: str = "data/stimuli_manifest.json",
          gt_dir: Path = Path("gt"), method: str = "smacof",
          expect_n_subjects: int = 41) -> Path:
    """Fit and save one extra GT. Returns the path written.

    Uses the same subject set as stage 1 - ``cohorts=("pilot",)`` by default plus
    ``variants=("pre",)`` - because a GT is a geometry over a *stimulus set*, and the post-SHINE half
    judged different images. The count is asserted for the same reason stage 1 asserts it: a silently
    different subject set would make this GT incomparable with the one the scan was run on.
    """
    from SpAM_Simulations.gt_construction import build_gt
    from SpAM_Simulations.pilot import load_pilot_subjects

    # The pilot data is deliberately NOT on the box between runs: every EC2 entrypoint's exit trap
    # does `rm -rf "$PILOT_DIR"` so human-subjects data never survives a run, let alone a
    # terminated instance. So any in-clone follow-up has to re-fetch it, and the bare
    # FileNotFoundError from the manifest loader does not say that.
    if not Path(manifest).is_file():
        raise SystemExit(
            f"{manifest} not found. The pilot data is scrubbed at the end of every run by the exit "
            f"trap, so a follow-up in an existing clone has to re-fetch it:\n"
            f"    aws s3 sync \"$S3_URI/data\" {pilot_dir}/ --only-show-errors\n"
            f"and should scrub it again afterwards:\n"
            f"    rm -rf {pilot_dir}"
        )
    subjects = load_pilot_subjects(pilot_dir, manifest, variants=("pre",))
    if len(subjects) != expect_n_subjects:
        raise SystemExit(
            f"expected {expect_n_subjects} pre-SHINE pilot subjects, got {len(subjects)}. This GT "
            f"must be built on the same set as the stage-1 scan or it cannot be compared with it."
        )
    print(f"[extra-gt] {len(subjects)} pre-SHINE pilot subjects; fitting ndim={ndim} ...",
          flush=True)
    coords, info = build_gt(subjects, ndim, method=method)

    gt_dir.mkdir(parents=True, exist_ok=True)
    out = gt_dir / f"gt_pre_shine_d{ndim}.npy"
    np.save(out, coords)

    # Append rather than overwrite: several extra dimensionalities may be built over time.
    notes_path = gt_dir / "extra_gts.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {"built": []}
    notes["built"] = [b for b in notes["built"] if b.get("n_dims") != int(ndim)]
    notes["built"].append({
        "n_dims": int(ndim),
        "gt_file": out.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gt_info": info,
        "reason": (
            "The stage-1 scan selects the dimensionality at which two 20-subject halves still "
            "agree, which is a floor set by sample size rather than the space's intrinsic "
            "dimensionality. A higher-D GT is used for stage 2 so the planning simulation errs "
            "pessimistic rather than optimistic."
        ),
    })
    notes["built"].sort(key=lambda b: b["n_dims"])
    notes_path.write_text(json.dumps(notes, indent=2, default=str))

    print(f"[extra-gt] wrote {out} (shape {coords.shape}, observed {info['observed_frac']:.1%} "
          f"of pairs) and recorded it in {notes_path}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndim", type=int, required=True, help="dimensionality to fit")
    p.add_argument("--pilot-dir", default="data")
    p.add_argument("--manifest", default="data/stimuli_manifest.json")
    p.add_argument("--gt-dir", type=Path, default=Path("gt"))
    p.add_argument("--method", default="smacof", choices=("smacof", "classical"))
    p.add_argument("--expect-n-subjects", type=int, default=41)
    a = p.parse_args(argv)
    if a.ndim <= 0:
        raise SystemExit(f"--ndim must be positive, got {a.ndim}")
    build(a.ndim, a.pilot_dir, a.manifest, a.gt_dir, a.method, a.expect_n_subjects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
