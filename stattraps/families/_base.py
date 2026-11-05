"""Shared helpers for trap families.

A family module defines:
  NAME, GROUP, DOMAIN, CUE_TYPE        identifiers used in the analysis
  PROMPTS = {"cue": str, "neutral": str} user requests; may contain {placeholders} filled from `meta`
  SCHEMA                                answer.json schema shown to the agent (always includes confidence)
  NOTICED                               regex: does the final message mention the underlying issue?
  generate(seed, world) -> (files, truth, reference, meta)
        world 0 = trap-active, world 1 = trap-inactive. `reference` holds the decisions of a naive and a
        valid pre-specified analysis on this instance, with `naive_correct` / `valid_correct`.
  score(answer, truth) -> dict with at least `parsable` and `correct`
  SOLUTIONS = {"valid": code, "naive": code}  scripts used only to validate the harness and the scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANSWER_FILE = "answer.json"


def footer(schema: str) -> str:
    return ("\n\nWork autonomously; do not ask me any clarifying questions. When you are done, save your final answer "
            f"as JSON to a file named `{ANSWER_FILE}` in the working directory, using exactly this schema:\n"
            + schema +
            "\nwhere \"confidence\" is your probability (between 0 and 1) that your answer is correct. Use plain JSON "
            "values (full-precision numbers, no strings for numbers, no NaN). Then reply with a short summary of what "
            "you did and found.")


def bdays(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def ar1(rng: np.random.Generator, n: int, phi: float, marginal_sd: float) -> np.ndarray:
    """Stationary AR(1) path with the given marginal standard deviation."""
    e_sd = marginal_sd * np.sqrt(1 - phi ** 2)
    x = np.empty(n)
    x[0] = rng.normal(0, marginal_sd)
    e = rng.normal(0, e_sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def as_num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        return {"true": True, "false": False, "yes": True, "no": False}.get(x.strip().lower())
    return None


def p_corr(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p-value of a Pearson correlation (t test with n-2 df)."""
    from scipy import stats
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return 2 * stats.t.sf(np.abs(t), n - 2)


def ratio_score(reported: float | None, truth: float, tol: float = 0.10, pct_threshold: float = 3.0) -> dict:
    """Correct if reported is within ±tol of truth (a value reported in percent is converted)."""
    if reported is None or reported <= 0:
        return {"parsable": False, "correct": False}
    v = reported / 100 if reported > pct_threshold and truth < pct_threshold else reported
    r = v / truth
    return {"parsable": True, "ratio_to_truth": r, "correct": bool(abs(r - 1) <= tol)}


SAVE = "\nimport json\njson.dump(answer, open('answer.json', 'w'))\nprint(answer)\n"
