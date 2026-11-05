"""
Minimal data-analysis agent: an LLM with one tool, `run_python`, in a loop.

Kept deliberately small (~150 lines) so that results measure the MODEL, not an agent framework:
  - one neutral system prompt (no hints about any trap), optional skill appended for the "skill" condition
  - one tool executing Python in a persistent kernel whose working directory holds the user's files
  - OpenAI-compatible Chat Completions over plain HTTP; assistant messages are sent back exactly as
    received, so provider-specific fields (DeepSeek reasoning_content, Gemini thought signatures) survive
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "You are a data analysis assistant. You can run Python code with the `run_python` tool; it executes in a "
    "persistent session (variables are kept between calls) whose working directory contains the user's files. "
    "Available packages include pandas, numpy, scipy, statsmodels, scikit-learn, cvxpy and matplotlib. "
    "Print what you need to see. When you have finished, reply to the user without calling a tool."
)
TOOLS = [{"type": "function", "function": {
    "name": "run_python",
    "description": "Execute Python code in a persistent session. Returns everything printed (stdout/stderr) "
                   "and the value of a trailing expression.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code"}},
                   "required": ["code"]}}}]
MAX_TOOL_OUTPUT = 10000


class Kernel:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self._start()

    def _start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "stattraps.kernel"], cwd=self.workdir, env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=lambda p=self.p, q=self.q: [q.put(l) for l in p.stdout], daemon=True).start()

    def run(self, code: str, timeout: float) -> tuple[str, bool]:
        try:
            self.p.stdin.write(json.dumps({"code": code}) + "\n")
            self.p.stdin.flush()
            r = json.loads(self.q.get(timeout=timeout))
            return r["output"], r["error"]
        except queue.Empty:
            self.close()
            self._start()
            return f"TimeoutError: execution exceeded {timeout:.0f} s; the session was restarted (variables lost).", True
        except Exception as e:  # kernel died
            self.close()
            self._start()
            return f"KernelError: {e}; the session was restarted (variables lost).", True

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=10)   # on Windows the workspace cannot be deleted while the process lives
        except Exception:
            pass


def _truncate(s: str, n: int = MAX_TOOL_OUTPUT) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} characters omitted]...\n" + s[-n // 2:]


def chat(provider: dict, messages: list, state: dict) -> dict:
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {provider['api_key']}", "Content-Type": "application/json"}
    body = {"model": provider["model"], "messages": messages, "tools": TOOLS, **provider.get("params", {})}
    if state.get("strip_reasoning"):
        body["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
    delay = 5.0
    for attempt in range(8):
        if provider.get("_throttle"):
            provider["_throttle"]()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=600)
        except requests.RequestException as e:
            err = f"network: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code == 400 and "reasoning" in r.text.lower() and not state.get("strip_reasoning"):
                state["strip_reasoning"] = True  # provider rejects echoed reasoning: drop it and retry
                body["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
                continue
            if r.status_code == 429 and re.search(r"per ?day|daily|PerDay|exhausted your|insufficient", r.text, re.I):
                raise RuntimeError(f"DAILY_QUOTA {err}")   # waiting will not help today; the runner stops this model
            if r.status_code == 402:
                raise RuntimeError(f"OUT_OF_CREDIT {err}")
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise RuntimeError(err)
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up after retries: {err}")


def run_task(provider: dict, prompt: str, files: dict, workdir: Path, skill: str | None = None,
             max_steps: int = 25, code_timeout: float = 300) -> dict:
    """Run one task. `files` maps file name -> path to copy into the workspace."""
    workdir.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        (workdir / name).write_bytes(Path(src).read_bytes())
    system = SYSTEM_PROMPT + (f"\n\nFollow this checklist whenever it applies:\n\n{skill}" if skill else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}
    state, status, final_text, n_tools, n_err, t0 = {}, "max_steps", "", 0, 0, time.time()
    kernel = Kernel(workdir)
    try:
        for _ in range(max_steps):
            resp = chat(provider, messages, state)
            u = resp.get("usage") or {}
            usage["calls"] += 1
            usage["prompt_tokens"] += u.get("prompt_tokens") or 0
            usage["completion_tokens"] += u.get("completion_tokens") or 0
            usage["cached_prompt_tokens"] += (u.get("prompt_cache_hit_tokens")
                                              or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            msg = {k: v for k, v in resp["choices"][0]["message"].items() if v is not None}
            msg["role"] = "assistant"
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                final_text, status = msg.get("content") or "", "completed"
                break
            for c in calls:
                n_tools += 1
                try:
                    code = json.loads(c["function"].get("arguments") or "{}")["code"]
                    out, is_err = kernel.run(code, code_timeout)
                except (json.JSONDecodeError, KeyError, TypeError):
                    out, is_err = "Invalid tool arguments: expected JSON {\"code\": \"...\"}.", True
                n_err += is_err
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": _truncate(out) or "(no output)"})
    except Exception as e:
        status, final_text = "error", f"{type(e).__name__}: {e}"
    finally:
        kernel.close()
    ans_path = workdir / "answer.json"
    answer = None
    if ans_path.exists():
        try:
            answer = json.loads(ans_path.read_text(encoding="utf-8"))
        except Exception:
            answer = None
    return {"status": status, "final_text": final_text, "answer": answer, "usage": usage, "n_tool_calls": n_tools,
            "n_tool_errors": n_err, "duration_s": round(time.time() - t0, 1),
            "transcript": [{k: (_truncate(v, 4000) if isinstance(v, str) else v) for k, v in m.items()}
                           for m in messages[1:]]}
