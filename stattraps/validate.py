"""
Offline validation of the harness and the scoring - no real LLM, no API key.

A scripted fake model, served over a local OpenAI-compatible endpoint, answers every prompt by calling
`run_python` with a reference solution (the valid or the naive analysis of that family) and then replying.
Each run goes through the real agent loop (kernel, tool calls, answer.json, scoring). The check: the score
the pipeline assigns must equal the reference decision computed by the instance generator.

python -m stattraps.validate --instances instances [--workers 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from . import suite
from .agent import run_task

REGISTRY: dict[str, str] = {}   # prompt text -> family


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        variant = body["model"].split(":", 1)[1]          # "scripted:valid" / "scripted:naive" / "scripted:mixed-0.6"
        if variant.startswith("mixed"):                   # test double for the runner: valid with probability q
            import random
            variant = "valid" if random.random() < float(variant.split("-")[1]) else "naive"
        msgs = body["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        fam = REGISTRY[user]
        if not any(m["role"] == "tool" for m in msgs):
            code = suite.FAMILIES[fam].SOLUTIONS[variant]
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "run_python", "arguments": json.dumps({"code": code})}}]}
        else:
            msg = {"role": "assistant", "content": f"Done ({variant} analysis)."}
        out = json.dumps({"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                          "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--families", default=None)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    insts = [suite.load(d) for d in suite.instances(Path(a.instances), a.families.split(",") if a.families else None)]
    for inst in insts:
        for c in suite.PROMPT_CONDITIONS:
            REGISTRY[inst["prompts"][c]] = inst["meta"]["family"]
    tmp = Path(tempfile.mkdtemp(prefix="stattraps_validate_"))

    def one(job):
        inst, variant, cond = job
        prov = {"base_url": base, "model": f"scripted:{variant}", "api_key": "none"}
        rec = run_task(prov, inst["prompts"][cond], inst["files"], tmp / f"{inst['meta']['instance_id']}_{variant}_{cond}")
        s = suite.score(inst, rec["answer"], rec["final_text"])
        expected = bool(inst["reference"][f"{variant}_correct"])
        return {"instance": inst["meta"]["instance_id"], "family": inst["meta"]["family"], "world": inst["meta"]["world"],
                "variant": variant, "prompt": cond, "status": rec["status"], "n_tool_errors": rec["n_tool_errors"],
                "scored_correct": bool(s.get("correct")), "expected": expected, "match": bool(s.get("correct")) == expected}

    jobs = [(i, v, c) for i in insts for v in ("valid", "naive") for c in ("neutral",)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = pd.DataFrame(list(ex.map(one, jobs)))
    shutil.rmtree(tmp, ignore_errors=True)
    srv.shutdown()
    print(f"{len(res)} scripted runs through the real agent loop in {time.time() - t0:.0f}s")
    print(res.groupby(["family", "variant"]).agg(runs=("match", "size"), scored_correct=("scored_correct", "mean"),
                                                 matches_reference=("match", "mean"),
                                                 tool_errors=("n_tool_errors", "sum")).round(2).to_string())
    bad = res[~res.match | (res.status != "completed")]
    if len(bad):
        print("\nMISMATCHES / FAILED RUNS:\n", bad.to_string())
        raise SystemExit(1)
    print("\nOK: every pipeline score equals the generator's reference decision.")


if __name__ == "__main__":
    main()
