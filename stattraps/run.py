"""
Run models on the StatTraps instances.

python -m stattraps.run --models configs/models.json --use deepseek-v4-pro,gemini-flash \
    --instances instances --prompts neutral,cue --k 2 --budget-usd 3 --workers 4

- API keys come from environment variables named in the model config (a `.env` file in the working
  directory is read automatically; never commit it).
- Output: results/<model>/<prompt>[_skill]/<instance>/rep_XX.json (raw records; scoring happens in analyze).
- Order: repetition-major and interleaved across families/worlds/prompts, so a run stopped early (budget,
  Ctrl+C) still leaves a balanced design. Resumable: finished runs are skipped; runs that never reached the
  model are redone.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import threading
import time
from pathlib import Path

from . import suite
from .agent import run_task

HARNESS = "minimal-v1"
SKILL = (Path(__file__).parent / "skill.md").read_text(encoding="utf-8")


def load_dotenv(path=".env"):
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def in_peak(price: dict | None) -> bool:
    pk = (price or {}).get("peak")
    if not pk:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    return (not pk.get("weekdays_only", True) or now.weekday() < 5) and any(a <= now.hour < b for a, b in pk["utc_hours"])


def run_cost(rec: dict, price: dict | None) -> float:
    """Estimated USD cost of one run (peak multiplier applied if the provider is in its peak hours now)."""
    u = rec.get("usage") or {}
    if not price or not u:
        return 0.0
    mult = price["peak"]["multiplier"] if in_peak(price) else 1.0
    price = {k: v * mult for k, v in price.items() if isinstance(v, (int, float))}
    cached = min(u.get("cached_prompt_tokens", 0), u.get("prompt_tokens", 0))
    return ((u.get("prompt_tokens", 0) - cached) * price.get("input", 0) + cached * price.get("cached_input",
            price.get("input", 0)) + u.get("completion_tokens", 0) * price.get("output", 0)) / 1e6


def infra_failure(rec: dict) -> bool:
    """An API/network/quota error is never data about the model: the run is not saved and is redone later."""
    return rec.get("status") == "error"


def make_throttle(rpm: float | None):
    if not rpm:
        return None
    lock, last = threading.Lock(), [0.0]

    def wait():
        with lock:
            gap = 60.0 / rpm - (time.time() - last[0])
            if gap > 0:
                time.sleep(gap)
            last[0] = time.time()
    return wait


def run_model(name: str, cfg: dict, insts: list[dict], a) -> None:
    key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not key:
        print(f"[{name}] SKIPPED: environment variable {cfg.get('api_key_env')} is not set", flush=True)
        return
    if str(cfg.get("model", "")).startswith("REPLACE"):
        print(f"[{name}] SKIPPED: set the model id in {a.models} (see: python -m stattraps.models --list {name})", flush=True)
        return
    if a.offpeak_only:
        while in_peak(cfg.get("price")):
            print(f"[{name}] peak-price hours now ({dt.datetime.now(dt.timezone.utc):%H:%M} UTC); waiting for "
                  f"off-peak... (Ctrl+C to stop)", flush=True)
            time.sleep(300)
    prov = {"base_url": cfg["base_url"], "model": cfg["model"], "api_key": key, "params": cfg.get("params", {}),
            "_throttle": make_throttle(cfg.get("rpm"))}
    price = cfg.get("price")
    out = Path(a.out) / name
    cond_suffix = "_skill" if a.skill else ""
    jobs = []
    for rep in range(a.k):
        for inst in insts:
            for p in a.prompts:
                dest = out / f"{p}{cond_suffix}" / inst["meta"]["instance_id"] / f"rep_{rep:02d}.json"
                if dest.exists():
                    try:
                        if not infra_failure(json.loads(dest.read_text(encoding="utf-8"))):
                            continue
                    except Exception:
                        pass
                jobs.append((inst, p, rep, dest))
    spent = [sum(run_cost(json.loads(f.read_text(encoding="utf-8")), price) for f in out.glob("*/*/rep_*.json"))]
    print(f"[{name}] {len(jobs)} runs to do; spent so far ~${spent[0]:.3f}"
          + (f" of ${a.budget_usd:.2f}" if a.budget_usd else ""), flush=True)
    stop = threading.Event()
    lock = threading.Lock()

    def one(job):
        inst, p, rep, dest = job
        if stop.is_set():
            return
        work = Path(a.out) / "_work" / f"{name}_{inst['meta']['instance_id']}_{p}{cond_suffix}_{rep}"
        shutil.rmtree(work, ignore_errors=True)
        rec = run_task(prov, inst["prompts"][p], inst["files"], work, skill=SKILL if a.skill else None,
                       max_steps=a.max_steps, code_timeout=a.code_timeout)
        shutil.rmtree(work, ignore_errors=True)
        rec.update({"model": name, "provider_model": cfg["model"], "harness": HARNESS,
                    "instance_id": inst["meta"]["instance_id"], "family": inst["meta"]["family"],
                    "world": inst["meta"]["world"], "prompt": p, "skill": bool(a.skill), "rep": rep,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        if infra_failure(rec):
            msg = rec["final_text"]
            print(f"[{name}] {inst['meta']['instance_id']} {p} rep {rep}: NOT SAVED (API error: {msg[:200]})", flush=True)
            if "DAILY_QUOTA" in msg or "OUT_OF_CREDIT" in msg:
                if not stop.is_set():
                    print(f"[{name}] {'Daily quota reached' if 'DAILY' in msg else 'Out of credit'} - stopping this "
                          f"model. Run the same command again later; finished runs are kept.", flush=True)
                stop.set()
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        s = suite.score(inst, rec["answer"], rec["final_text"])
        with lock:
            spent[0] += run_cost(rec, price)
            if a.budget_usd and spent[0] >= a.budget_usd:
                stop.set()
        print(f"[{name}] {inst['meta']['instance_id']} {p} rep {rep}: {rec['status']}, "
              f"{'correct' if s.get('correct') else ('WRONG' if s.get('parsable') else 'no answer')}, "
              f"{rec['duration_s']}s, ~${spent[0]:.3f}", flush=True)
        if stop.is_set():
            print(f"[{name}] BUDGET REACHED - no new runs.", flush=True)

    with cf.ThreadPoolExecutor(cfg.get("workers", a.workers)) as ex:
        list(ex.map(one, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.json")
    ap.add_argument("--use", required=True, help="comma-separated model names from the config")
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--families", default=None)
    ap.add_argument("--prompts", default="neutral,cue")
    ap.add_argument("--k", type=int, default=2, help="repetitions per instance and prompt")
    ap.add_argument("--skill", action="store_true", help="append the statistical checklist to the system prompt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget-usd", type=float, default=None, help="per model")
    ap.add_argument("--workers", type=int, default=4, help="parallel runs per model")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--code-timeout", type=float, default=300)
    ap.add_argument("--offpeak-only", action="store_true",
                    help="for providers with a 'peak' price entry, wait until off-peak hours before starting")
    a = ap.parse_args()
    a.prompts = a.prompts.split(",")
    load_dotenv()
    cfgs = json.loads(Path(a.models).read_text(encoding="utf-8"))
    paths = suite.instances(Path(a.instances), a.families.split(",") if a.families else None)
    insts = [suite.load(p) for p in paths]
    insts.sort(key=lambda i: (i["meta"]["index"], i["meta"]["world"], list(suite.FAMILIES).index(i["meta"]["family"])))
    names = a.use.split(",")
    missing = [n for n in names if n not in cfgs]
    if missing:
        raise SystemExit(f"not in {a.models}: {missing}")
    print(f"{len(insts)} instances x {len(a.prompts)} prompts x k={a.k} for {names}", flush=True)
    threads = [threading.Thread(target=run_model, args=(n, cfgs[n], insts, a)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
