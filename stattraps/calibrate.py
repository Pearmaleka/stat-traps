"""
Trap-strength calibration: how would a careless and a careful pre-specified analysis score?

For each family and world, generate N fresh instances (seeds disjoint from the benchmark) and report the
share of instances where the naive analysis and the valid analysis reach the correct answer. A good trap has
naive ~50% or less in w0, ~100% in w1, and valid close to the nominal level in both.

python -m stattraps.calibrate --n 100 [--families SPUR,DEP]
"""
import argparse
import time

import pandas as pd

from .suite import FAMILIES

CAL_SEED = 777_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--families", default=None)
    ap.add_argument("--out", default="calibration.csv")
    a = ap.parse_args()
    rows = []
    for f_idx, fam in enumerate(a.families.split(",") if a.families else FAMILIES):
        mod = FAMILIES[fam]
        t0 = time.time()
        for world in (0, 1):
            for i in range(a.n):
                _, _, ref, _ = mod.generate(CAL_SEED + 1_000_000 * f_idx + 10_000 * world + i, world)
                rows.append({"family": fam, "world": world, "naive_correct": bool(ref["naive_correct"]),
                             "valid_correct": bool(ref["valid_correct"]),
                             **{k: v for k, v in ref.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        print(f"{fam}: {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    tab = df.groupby(["family", "world"])[["naive_correct", "valid_correct"]].mean().unstack("world")
    tab.columns = [f"{m}_w{w}" for m, w in tab.columns]
    tab["naive_balanced"] = (tab.naive_correct_w0 + tab.naive_correct_w1) / 2
    tab["valid_balanced"] = (tab.valid_correct_w0 + tab.valid_correct_w1) / 2
    print(f"\nShare of instances answered correctly (n = {a.n} per world)\n")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
