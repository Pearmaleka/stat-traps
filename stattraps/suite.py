"""Build, load and score the StatTraps suite.

Instance id: <FAMILY>_w<world>_<index>, e.g. SPUR_w0_03. Each instance directory holds the data files the agent
sees, plus prompts.json (cue / neutral), truth.json, reference.json and meta.json, which the agent never sees.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .families import dep, freq, gaps, leak, pseudo, simpson, snoop, split, spur
from .families._base import footer

FAMILIES = {m.NAME: m for m in (spur, dep, pseudo, snoop, simpson, leak, freq, split, gaps)}
PROMPT_CONDITIONS = ("neutral", "cue")
BASE_SEED = 20261105
AGENT_FILES = ("prompts.json", "truth.json", "reference.json", "meta.json")   # never copied to the agent


def seed_for(family: str, world: int, index: int, base_seed: int = BASE_SEED) -> int:
    return base_seed + 100_000 * list(FAMILIES).index(family) + 1_000 * world + index


def build(root: Path, n_per_world: int = 5, families=None, base_seed: int = BASE_SEED, verbose=True) -> list[Path]:
    root = Path(root)
    out = []
    for fam in families or FAMILIES:
        mod = FAMILIES[fam]
        for world in (0, 1):
            for i in range(n_per_world):
                seed = seed_for(fam, world, i, base_seed)
                d = root / f"{fam}_w{world}_{i:02d}"
                if (d / "meta.json").exists():
                    out.append(d)
                    continue
                files, truth, ref, extra = mod.generate(seed, world)
                d.mkdir(parents=True, exist_ok=True)
                for name, obj in files.items():
                    if isinstance(obj, pd.DataFrame):
                        obj.to_csv(d / name, index=False)
                    else:
                        (d / name).write_text(obj, encoding="utf-8")
                prompts = {c: mod.PROMPTS[c].format(**extra) + footer(mod.SCHEMA) for c in PROMPT_CONDITIONS}
                meta = {"instance_id": d.name, "family": fam, "world": world, "index": i, "seed": seed,
                        "group": mod.GROUP, "domain": mod.DOMAIN, "cue_type": mod.CUE_TYPE, **extra}
                for name, obj in (("prompts.json", prompts), ("truth.json", truth), ("reference.json", ref),
                                  ("meta.json", meta)):
                    (d / name).write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
                out.append(d)
                if verbose:
                    print(f"  {d.name}", flush=True)
    return out


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except Exception:
        pass
    return str(o)


def load(d: Path) -> dict:
    d = Path(d)
    j = lambda n: json.loads((d / n).read_text(encoding="utf-8"))
    return {"dir": d, "meta": j("meta.json"), "prompts": j("prompts.json"), "truth": j("truth.json"),
            "reference": j("reference.json"),
            "files": {p.name: p for p in d.iterdir() if p.is_file() and p.name not in AGENT_FILES}}


def instances(root: Path, families=None) -> list[Path]:
    fams = set(families) if families else None
    return sorted(p for p in Path(root).iterdir() if (p / "meta.json").exists()
                  and (fams is None or p.name.split("_")[0] in fams))


def score(inst: dict, answer, final_text: str = "") -> dict:
    mod = FAMILIES[inst["meta"]["family"]]
    s = mod.score(answer, inst["truth"]) if isinstance(answer, dict) else {"parsable": False, "correct": False}
    conf = answer.get("confidence") if isinstance(answer, dict) else None
    try:
        conf = float(conf)
        conf = conf / 100 if conf > 1 else conf
        conf = conf if 0 <= conf <= 1 else None
    except (TypeError, ValueError):
        conf = None
    s["confidence"] = conf
    s["noticed"] = bool(re.search(mod.NOTICED, final_text or "", re.I))
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the StatTraps instances")
    ap.add_argument("--out", default="instances")
    ap.add_argument("--n-per-world", type=int, default=5)
    ap.add_argument("--families", default=None, help="comma-separated subset")
    a = ap.parse_args()
    paths = build(Path(a.out), a.n_per_world, a.families.split(",") if a.families else None)
    print(f"{len(paths)} instances in {a.out}/")
