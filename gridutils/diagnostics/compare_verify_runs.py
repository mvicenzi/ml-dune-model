"""
compare_verify_runs.py
──────────────────────
Read out the verification runs for the objective switch.

Three questions, in the order they matter:

1. **Did the default change?** `verify_main` and `verify_branch` are the same config, same
   seed, on the two code trees. The branch states `objective: "hybrid"` explicitly; main
   predates the key.

   A difference between them is only meaningful next to a difference the code change
   cannot explain, so `verify_main2` runs the reference a second time -- same code, same
   seed, same everything -- and its disagreement with `verify_main` is the floor. Judging
   against a borrowed number would not do: docs/17's 2.6e-2 measured drift in backbone
   FEATURES, which is not the same quantity as a loss and not interchangeable with it.

2. **Does mae train?** Both terms present, both moving, and a checkpoint written whose key
   set has no teacher in it.

3. **Does mae work on two ranks?** Same, plus the run has to finish -- an unbalanced
   collective would hang rather than fail, so completion is itself the result.

Run:  python -u gridutils/diagnostics/compare_verify_runs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get("CONDOR_OUT",
                          "/gpfs01/lbne/users/fm/mvicenzi/CONDOR_OUT"))


def load_histories(run: str) -> dict | None:
    for p in (OUT / run / "histories.json", OUT / run / run / "histories.json"):
        if p.exists():
            return json.loads(p.read_text())
    hits = list((OUT / run).rglob("histories.json"))
    return json.loads(hits[0].read_text()) if hits else None


def checkpoints(run: str) -> list[Path]:
    return sorted((OUT / run).rglob("checkpoint_epoch*.pt"))


def finished(run: str) -> tuple[bool, str]:
    """Did the job reach the end, and what did it last say?"""
    outs = sorted((OUT / run).glob("*.out"))
    if not outs:
        return False, "no .out file"
    text = outs[-1].read_text(errors="replace")
    tail = [ln for ln in text.strip().splitlines() if ln.strip()][-1:] or ["(empty)"]
    return ("[timing] epoch=" in text), tail[0][:110]


def series(h: dict, key: str) -> list[float]:
    return [v for v in (h or {}).get(key, []) if v is not None]


def describe(name: str, h: dict | None) -> None:
    ok, tail = finished(name)
    ck = checkpoints(name)
    print(f"\n{name}")
    print(f"  reached epoch end : {ok}")
    print(f"  last line         : {tail}")
    print(f"  checkpoints       : {[p.name for p in ck] or 'none'}")
    if not h:
        print("  histories.json    : MISSING")
        return
    loss = series(h, "loss")
    print(f"  iterations logged : {len(loss)}")
    if loss:
        print(f"  loss first -> last: {loss[0]:.6f} -> {loss[-1]:.6f}")
    for key in ("loss_charge", "loss_occ", "loss_masked", "loss_unmasked", "kl"):
        s = series(h, key)
        if s:
            print(f"  {key:18s}: {s[0]:.6f} -> {s[-1]:.6f}  ({len(s)} pts)")


def loss_diff(a: str, b: str):
    """(n, exact, max abs, mean relative) between two runs' loss curves, or None."""
    ha, hb = load_histories(a), load_histories(b)
    if not (ha and hb):
        return None
    la, lb = series(ha, "loss"), series(hb, "loss")
    n = min(len(la), len(lb))
    if n == 0:
        return None
    diffs = [abs(x - y) for x, y in zip(la[:n], lb[:n])]
    rel = [d / max(abs(x), 1e-12) for d, x in zip(diffs, la[:n])]
    return n, sum(1 for d in diffs if d == 0.0), max(diffs), sum(rel) / n, la, lb


def compare_pair() -> None:
    """Judge the code change against the stack's own run-to-run noise.

    Two runs of the SAME code with the same seed are not expected to agree bit for bit
    on this stack -- sparse scatter uses atomics and the kernel autotuner picks by
    timing. So "not identical" says nothing on its own. What matters is whether the
    change moves the loss further than re-running the reference does, which is why
    verify_main2 exists: it is verify_main again, identical in every way.
    """
    print("\n" + "=" * 72)
    print("1. DID THE DEFAULT CHANGE?")
    print("=" * 72)

    floor = loss_diff("verify_main", "verify_main2")
    test = loss_diff("verify_main", "verify_branch")

    if test is None:
        print("   cannot compare: a histories.json is missing")
        return

    n, exact, mx, mrel, la, lb = test
    print(f"\n   main vs branch   (the code change)")
    print(f"     iterations       : {n}")
    print(f"     bit-identical    : {exact}/{n}")
    print(f"     max |difference| : {mx:.3e}")
    print(f"     mean relative    : {mrel:.3e}")
    print(f"     first / last     : {la[0]:.9f} vs {lb[0]:.9f}  |  "
          f"{la[n-1]:.9f} vs {lb[n-1]:.9f}")

    if floor is None:
        print("\n   main vs main2    : NOT AVAILABLE -- no floor to judge against.")
        print("   Without it, the numbers above cannot be called pass or fail.")
        return

    fn, fexact, fmx, fmrel, _, _ = floor
    print(f"\n   main vs main2    (the same code, twice -- the noise floor)")
    print(f"     iterations       : {fn}")
    print(f"     bit-identical    : {fexact}/{fn}")
    print(f"     max |difference| : {fmx:.3e}")
    print(f"     mean relative    : {fmrel:.3e}")

    print()
    if exact == n:
        print("   VERDICT: bit-identical. The objective switch is inert on the default path.")
    elif fexact == fn:
        print("   VERDICT: the reference reproduces itself exactly, so this stack IS")
        print("   deterministic -- and the change is not. Something in the default path")
        print("   moved. Investigate before trusting any later comparison.")
    else:
        ratio = mrel / max(fmrel, 1e-30)
        print(f"   Neither pair is bit-identical, so the stack is nondeterministic at")
        print(f"   ~{fmrel:.1e} mean relative. The change sits at {mrel:.1e}, a factor")
        print(f"   {ratio:.2f} of that.")
        if ratio <= 3.0:
            print("   VERDICT: within the stack's own run-to-run spread. No evidence the")
            print("   default path changed.")
        else:
            print("   VERDICT: larger than run-to-run noise. Treat the default path as")
            print("   changed until explained.")


def main() -> int:
    compare_pair()

    print("\n" + "=" * 72)
    print("2. DOES mae TRAIN?")
    print("=" * 72)
    for run in ("verify_mae", "verify_mae_ddp2"):
        describe(run, load_histories(run))

    print("\n" + "=" * 72)
    print("3. CHECKPOINT SHAPE (mae must carry no teacher)")
    print("=" * 72)
    try:
        import torch
        from dino.config import DINOConfig
        for run in ("verify_mae", "verify_mae_ddp2", "verify_branch"):
            ck = checkpoints(run)
            if not ck:
                print(f"  {run:18s} no checkpoint")
                continue
            with torch.serialization.safe_globals([DINOConfig]):
                d = torch.load(ck[-1], map_location="cpu")
            keys = sorted(k for k in d if not k.startswith("_"))
            recon = [k for k in d.get("student", {})
                     if k.startswith(("charge_head", "occ_coarse_block",
                                      "occupancy_head_coarse"))]
            print(f"  {run:18s} objective={d.get('objective', '(absent)')!r:10s} "
                  f"keys={keys}")
            print(f"  {'':18s} teacher present={'teacher' in d}  "
                  f"recon tensors={len(recon)}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  could not inspect checkpoints: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
