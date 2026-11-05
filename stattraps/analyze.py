"""
Score all runs and produce the tables and figures of the design (docs/DESIGN.md, section 7).

python -m stattraps.analyze --results results --instances instances --out report

Outputs in --out: runs_scored.csv, summary.md, and figures/*.png.
Confidence intervals: instance-clustered bootstrap (runs of one instance are not independent).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import suite

DECISION_FAMILIES = ["SPUR", "DEP", "PSEUDO", "SNOOP"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, BASE, SURF, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#898781"


# ----------------------------------------------------------------------------- scoring
def score_all(results: Path, instances: Path) -> pd.DataFrame:
    cache, rows = {}, []
    for f in sorted(results.glob("*/*/*/rep_*.json")):
        if "_work" in f.parts:
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") == "error":      # API/quota failure, not a model answer
            continue
        iid = rec["instance_id"]
        if iid not in cache:
            cache[iid] = suite.load(instances / iid)
        inst = cache[iid]
        s = suite.score(inst, rec.get("answer"), rec.get("final_text", ""))
        u = rec.get("usage") or {}
        rows.append({
            "model": rec["model"], "harness": rec.get("harness", "?"), "instance_id": iid,
            "family": inst["meta"]["family"], "group": inst["meta"]["group"], "cue_type": inst["meta"]["cue_type"],
            "world": inst["meta"]["world"], "prompt": rec["prompt"], "skill": bool(rec.get("skill")),
            "rep": rec["rep"], "status": rec["status"], "parsable": bool(s.get("parsable")),
            "correct": bool(s.get("correct")), "decision": s.get("decision"), "confidence": s.get("confidence"),
            "noticed": s.get("noticed"), "used_leak": s.get("used_leak"),
            "naive_correct_ref": bool(inst["reference"]["naive_correct"]),
            "valid_correct_ref": bool(inst["reference"]["valid_correct"]),
            "n_tool_calls": rec.get("n_tool_calls"), "n_tool_errors": rec.get("n_tool_errors"),
            "duration_s": rec.get("duration_s"), "calls": u.get("calls"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "final_text": (rec.get("final_text") or "")[-2000:],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def boot(df: pd.DataFrame, stat, B=2000, seed=0):
    """Instance-clustered bootstrap: resample instances, keep all their runs."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("instance_id")]
    point = stat(df)
    vals = []
    for _ in range(B):
        idx = rng.integers(len(groups), size=len(groups))
        v = stat(pd.concat([groups[i] for i in idx]))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def balanced_acc(d: pd.DataFrame) -> float:
    w = d.groupby("world").correct.mean()
    return float(w.mean()) if len(w) == 2 else np.nan


def auroc(score, label) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def ece(conf, correct, bins=10) -> float:
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi) if hi < 1 else (c >= lo) & (c <= hi)
        if m.any():
            e += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(e)


def fmt(p, lo=None, hi=None, pct=True):
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "–"
    return f(p) if lo is None else f"{f(p)} [{f(lo)}, {f(hi)}]"


# ----------------------------------------------------------------------------- tables
def tables(df: pd.DataFrame) -> str:
    out = []
    base = df[(df.prompt == "neutral") & (~df.skill)]
    models = sorted(df.model.unique())

    out.append("## 1. Headline: balanced accuracy, neutral prompt\n")
    out.append("Mean over families of (accuracy in trap-active world + accuracy in trap-inactive world) / 2. "
               "Reference rows: a pre-specified naive and valid analysis on the same instances.\n")
    rows = []
    for m in models:
        d = base[base.model == m]
        stat = lambda x: np.nanmean([balanced_acc(g) for _, g in x.groupby("family")])
        p, lo, hi = boot(d, stat)
        rows.append({"model": m, "balanced accuracy [95% CI]": fmt(p, lo, hi), "runs": len(d),
                     "trap-active acc": fmt(d[d.world == 0].correct.mean()),
                     "trap-inactive acc": fmt(d[d.world == 1].correct.mean())})
    inst = base.drop_duplicates("instance_id")
    for ref, col in (("naive analysis (reference)", "naive_correct_ref"), ("valid analysis (reference)", "valid_correct_ref")):
        bal = np.nanmean([g.groupby("world")[col].mean().mean() for _, g in inst.groupby("family")])
        rows.append({"model": ref, "balanced accuracy [95% CI]": fmt(bal), "runs": "–",
                     "trap-active acc": fmt(inst[inst.world == 0][col].mean()),
                     "trap-inactive acc": fmt(inst[inst.world == 1][col].mean())})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("## 2. Accuracy by family (neutral prompt): trap-active / trap-inactive\n")
    t = base.groupby(["family", "model", "world"]).correct.mean().unstack(["model", "world"])
    fam_rows = []
    for fam in suite.FAMILIES:
        if fam not in t.index:
            continue
        r = {"family": fam}
        for m in models:
            a0 = t.loc[fam].get((m, 0), np.nan)
            a1 = t.loc[fam].get((m, 1), np.nan)
            r[m] = f"{fmt(a0)} / {fmt(a1)}"
        g = inst[inst.family == fam]
        r["naive ref"] = f"{fmt(g[g.world == 0].naive_correct_ref.mean())} / {fmt(g[g.world == 1].naive_correct_ref.mean())}"
        r["valid ref"] = f"{fmt(g[g.world == 0].valid_correct_ref.mean())} / {fmt(g[g.world == 1].valid_correct_ref.mean())}"
        fam_rows.append(r)
    out.append(pd.DataFrame(fam_rows).to_markdown(index=False) + "\n")

    out.append("## 3. Agent as a statistical test (neutral prompt)\n")
    out.append("False-positive rate = claims an effect in the trap-active (null) world; power = claims it in the "
               "signal world.\n")
    rows = []
    for fam in DECISION_FAMILIES:
        d = base[base.family == fam]
        if d.empty:
            continue
        for m in models:
            x = d[(d.model == m) & d.parsable]
            dec = x.decision.astype(bool)
            rows.append({"family": fam, "who": m, "false-positive rate": fmt(dec[x.world == 0].mean()),
                         "power": fmt(dec[x.world == 1].mean())})
        g = inst[inst.family == fam]
        for who, col in (("naive test", "naive_correct_ref"), ("valid test", "valid_correct_ref")):
            rows.append({"family": fam, "who": who, "false-positive rate": fmt(1 - g[g.world == 0][col].mean()),
                         "power": fmt(g[g.world == 1][col].mean())})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("## 4. Effect of the user's cue (accuracy cue − neutral, trap-active world)\n")
    nos = df[~df.skill]
    rows = []
    for fam in suite.FAMILIES:
        for m in models:
            d = nos[(nos.family == fam) & (nos.model == m) & (nos.world == 0)]
            if d.prompt.nunique() < 2:
                continue
            per = d.groupby(["instance_id", "prompt"]).correct.mean().unstack()
            diff = (per["cue"] - per["neutral"]).dropna()
            from scipy import stats
            pv = stats.wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) > 1 else 1.0
            rows.append({"family": fam, "cue type": suite.FAMILIES[fam].CUE_TYPE, "model": m,
                         "neutral": fmt(per["neutral"].mean()), "cue": fmt(per["cue"].mean()),
                         "difference": f"{100 * diff.mean():+.0f} pts", "Wilcoxon p": f"{pv:.3g}"})
    out.append((pd.DataFrame(rows).to_markdown(index=False) if rows else "_cue condition not run_") + "\n")

    out.append("## 5. Awareness and confidence (neutral prompt)\n")
    rows = []
    for m in models:
        d = base[base.model == m]
        d0 = d[d.world == 0]
        c = d[d.confidence.notna()]
        rows.append({"model": m,
                     "noticed the issue (trap-active)": fmt(d0.noticed.mean()),
                     "noticed but still wrong (trap-active)": fmt((d0.noticed & ~d0.correct).mean()),
                     "confidence AUROC (all)": fmt(auroc(c.confidence, c.correct), pct=False),
                     "mean confidence when right / wrong": f"{fmt(c[c.correct].confidence.mean())} / "
                                                           f"{fmt(c[~c.correct].confidence.mean())}",
                     "ECE": fmt(ece(c.confidence, c.correct), pct=False)})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("## 6. Are errors systematic? (neutral prompt)\n")
    rows = []
    for m in models:
        d = base[base.model == m]
        per = d.groupby("instance_id").correct.agg(["mean", "size"])
        wrong = d[~d.correct]
        sys_share = wrong.instance_id.map(per["mean"]).eq(0).mean() if len(wrong) else np.nan
        rows.append({"model": m, "instances always right": int((per["mean"] == 1).sum()),
                     "always wrong": int((per["mean"] == 0).sum()), "mixed": int(((per["mean"] > 0) & (per["mean"] < 1)).sum()),
                     "share of wrong runs that are systematic": fmt(sys_share)})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    sk = df[df.skill]
    if not sk.empty:
        out.append("## 7. Checklist (skill) effect, trap-active world\n")
        rows = []
        for (m, p), d in sk.groupby(["model", "prompt"]):
            b = df[(~df.skill) & (df.model == m) & (df.prompt == p)]
            for fam in suite.FAMILIES:
                x, y = d[(d.family == fam) & (d.world == 0)], b[(b.family == fam) & (b.world == 0)]
                if len(x) and len(y):
                    rows.append({"model": m, "prompt": p, "family": fam, "no skill": fmt(y.correct.mean()),
                                 "skill": fmt(x.correct.mean())})
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("## 8. Cost and effort\n")
    out.append(df.groupby("model").agg(runs=("correct", "size"), no_answer=("parsable", lambda s: f"{100 * (1 - s.mean()):.0f}%"),
                                       tool_calls=("n_tool_calls", "mean"), model_calls=("calls", "mean"),
                                       prompt_tokens=("prompt_tokens", "mean"), seconds=("duration_s", "median"))
               .round(1).to_markdown() + "\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_facecolor(SURF)
    ax.set_axisbelow(True)


def figures(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "axes.labelcolor": INK2, "figure.facecolor": SURF})
    out.mkdir(parents=True, exist_ok=True)
    base = df[(df.prompt == "neutral") & (~df.skill)]
    models = sorted(df.model.unique())
    fams = [f for f in suite.FAMILIES if f in set(base.family)]
    col = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}

    # F1: accuracy in the trap-active world by family, per model, with naive/valid reference ticks
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(fams) + 1.4))
    y = np.arange(len(fams))
    inst = base.drop_duplicates("instance_id")
    for j, m in enumerate(models):
        acc = [base[(base.model == m) & (base.family == f) & (base.world == 0)].correct.mean() for f in fams]
        ax.scatter(acc, y + (j - (len(models) - 1) / 2) * 0.14, s=46, color=col[m], label=m, zorder=3,
                   edgecolor=SURF, linewidth=1.5)
    nv = [inst[(inst.family == f) & (inst.world == 0)].naive_correct_ref.mean() for f in fams]
    vv = [inst[(inst.family == f) & (inst.world == 0)].valid_correct_ref.mean() for f in fams]
    ax.scatter(nv, y, marker="|", s=260, color=MUTED, label="naive analysis", zorder=2)
    ax.scatter(vv, y, marker="|", s=260, color=INK, label="valid analysis", zorder=2)
    ax.set_yticks(y, fams)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_title("Accuracy when the obvious analysis is invalid", loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, len(models) + 2), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out / "fig1_trap_accuracy.png", dpi=200)
    plt.close(fig)

    # F2: agent as a test - false-positive rate vs power
    dec = base[base.family.isin(DECISION_FAMILIES) & base.parsable]
    if not dec.empty:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        markers = dict(zip(DECISION_FAMILIES, ["o", "s", "^", "D"]))
        for m in models:
            for f in DECISION_FAMILIES:
                x = dec[(dec.model == m) & (dec.family == f)]
                if len(x):
                    ax.scatter(x[x.world == 0].decision.astype(bool).mean(), x[x.world == 1].decision.astype(bool).mean(),
                               color=col[m], marker=markers[f], s=55, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.axvline(0.05, color=BASE, lw=1, ls="--")
        ax.text(0.06, 0.03, "nominal 5%", color=INK2, fontsize=8)
        for m in models:
            ax.scatter([], [], color=col[m], label=m, s=40)
        for f in DECISION_FAMILIES:
            ax.scatter([], [], color=MUTED, marker=markers[f], label=f, s=40)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("False-positive rate (no effect exists)")
        ax.set_ylabel("Power (effect exists)")
        ax.set_title("The agent as a statistical test", loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        _style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "fig2_agent_as_test.png", dpi=200)
        plt.close(fig)

    # F3: cue effect in the trap-active world
    nos = df[~df.skill & (df.world == 0)]
    if nos.prompt.nunique() == 2:
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(fams) + 1.4))
        for j, m in enumerate(models):
            d = nos[nos.model == m]
            for i, f in enumerate(fams):
                a = d[(d.family == f) & (d.prompt == "neutral")].correct.mean()
                b = d[(d.family == f) & (d.prompt == "cue")].correct.mean()
                yy = i + (j - (len(models) - 1) / 2) * 0.14
                ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="->", color=col[m], lw=1.6))
                ax.scatter([a], [yy], color=col[m], s=18, zorder=3)
        for m in models:
            ax.plot([], [], color=col[m], label=m)
        ax.set_yticks(range(len(fams)), [f"{f} ({suite.FAMILIES[f].CUE_TYPE.replace('_', ' ')})" for f in fams])
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_title("Neutral prompt → prompt with the user's cue (trap-active world)", loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.08))
        fig.tight_layout()
        fig.savefig(out / "fig3_cue_effect.png", dpi=200)
        plt.close(fig)

    # F4: reliability diagram of stated confidence
    c = base[base.confidence.notna()]
    if not c.empty:
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        edges = np.linspace(0, 1, 6)
        for m in models:
            x = c[c.model == m]
            pts = [(x[(x.confidence >= lo) & (x.confidence <= hi)].confidence.mean(),
                    x[(x.confidence >= lo) & (x.confidence <= hi)].correct.mean(),
                    ((x.confidence >= lo) & (x.confidence <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
            pts = [p for p in pts if p[2] > 0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col[m], lw=2, marker="o", ms=5, label=m)
        ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Stated confidence")
        ax.set_ylabel("Share correct")
        ax.set_title("Is the stated confidence honest?", loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "fig4_calibration.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = score_all(Path(a.results), Path(a.instances))
    if df.empty:
        raise SystemExit("no runs found")
    df.to_csv(out / "runs_scored.csv", index=False)
    md = f"# StatTraps results\n\n{len(df)} runs, models: {', '.join(sorted(df.model.unique()))}\n\n" + tables(df)
    (out / "summary.md").write_text(md, encoding="utf-8")
    figures(df, out / "figures")
    print(md)


if __name__ == "__main__":
    main()
