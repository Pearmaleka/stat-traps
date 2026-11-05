"""
How should an evaluation budget be split between instances and repeated runs? (RQ5)

Model: correctness y_ij of run j on instance i, with instance-level accuracy p_i.
  Var(mean accuracy) = sigma2_instance / T + sigma2_run / (T * k)
where T = instances, k = runs per instance, sigma2_instance = Var(p_i), sigma2_run = E[p_i (1 - p_i)].
With cost c_I to obtain an instance and c_R per run, the cost-optimal number of runs per instance is the
two-stage sampling result  k* = sqrt( (c_I / c_R) * (sigma2_run / sigma2_instance) )   (Cochran, ch. 10).

python -m stattraps.allocation --scored report/runs_scored.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def components(d: pd.DataFrame) -> dict:
    """One-way random-effects ANOVA (method of moments) for binary correctness."""
    g = d.groupby("instance_id").correct.agg(["mean", "var", "size"])
    g = g[g["size"] >= 2]
    if len(g) < 3:
        return {}
    k = g["size"].mean()
    s2_run = float(g["var"].mean())                          # within-instance (run-to-run) variance
    s2_inst = max(float(g["mean"].var(ddof=1)) - s2_run / k, 0.0)
    total = s2_inst + s2_run
    return {"instances": len(g), "runs_per_instance": k, "sigma2_instance": s2_inst, "sigma2_run": s2_run,
            "icc": s2_inst / total if total > 0 else np.nan, "accuracy": float(d.correct.mean())}


def half_width(c: dict, T: int, k: int) -> float:
    return 1.96 * np.sqrt(c["sigma2_instance"] / T + c["sigma2_run"] / (T * k))


def k_star(c: dict, cost_ratio: float) -> float:
    if c["sigma2_instance"] <= 0:
        return np.inf
    return float(np.sqrt(cost_ratio * c["sigma2_run"] / c["sigma2_instance"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="report/runs_scored.csv")
    ap.add_argument("--budget-runs", type=int, default=360)
    a = ap.parse_args()
    df = pd.read_csv(a.scored)
    df = df[~df.skill.astype(bool)]
    rows = []
    for (m, p), d in df.groupby(["model", "prompt"]):
        c = components(d)
        if not c:
            continue
        row = {"model": m, "prompt": p, **{k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()}}
        for ratio in (0, 10, 100):
            row[f"k* (instance costs {ratio}x a run)"] = round(k_star(c, ratio), 2) if ratio else "1 (generated instances)"
        B = a.budget_runs
        for k in (1, 2, 4):
            row[f"±CI with {B} runs, k={k}"] = round(half_width(c, B // k, k), 4)
        rows.append(row)
    if not rows:
        raise SystemExit("need >= 2 runs per instance for >= 3 instances")
    out = pd.DataFrame(rows)
    print(out.to_markdown(index=False))
    print("\nReading: icc = share of the variance in correctness that is between instances (systematic). "
          "If icc is high, repeating runs adds little; spend the budget on new instances.")


if __name__ == "__main__":
    main()
