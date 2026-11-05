"""
List the model ids a provider in your config offers (so you never have to guess them), or send one test
request to check that a key and model work.

python -m stattraps.models --list deepseek-v4-pro
python -m stattraps.models --ping deepseek-v4-pro
"""
import argparse
import json
import os
from pathlib import Path

import requests

from .run import load_dotenv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/models.json")
    ap.add_argument("--list", metavar="NAME", help="list model ids at this entry's base_url")
    ap.add_argument("--ping", metavar="NAME", help="send one tiny chat request with tools")
    a = ap.parse_args()
    load_dotenv()
    cfgs = json.loads(Path(a.models).read_text(encoding="utf-8"))
    name = a.list or a.ping
    cfg = cfgs[name]
    key = os.environ.get(cfg["api_key_env"], "")
    if not key:
        raise SystemExit(f"set {cfg['api_key_env']} in .env first")
    h = {"Authorization": f"Bearer {key}"}
    if a.list:
        r = requests.get(cfg["base_url"].rstrip("/") + "/models", headers=h, timeout=60)
        r.raise_for_status()
        for m in sorted(x.get("id", "") for x in r.json().get("data", [])):
            print(m)
    else:
        from .agent import TOOLS
        body = {"model": cfg["model"], "messages": [{"role": "user", "content": "Use run_python to print 2+2."}],
                "tools": TOOLS, **cfg.get("params", {})}
        r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", headers=h, json=body, timeout=120)
        print(r.status_code)
        print(json.dumps(r.json(), indent=1)[:1500])


if __name__ == "__main__":
    main()
