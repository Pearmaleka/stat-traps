"""SNOOP — data snooping / multiple testing among candidate trading signals.

40 candidate signals and the next-day return of a stock over 1,000 days (already aligned, so there is no
look-ahead to handle). The question is whether ANY signal has genuine predictive power.
w0: no signal is predictive. Testing the best of 40 at 5% finds "significance" ~87% of the time.
w1: one signal is truly predictive (correlation 0.16), detectable after a family-wise correction.
Valid: Bonferroni / Holm (or a held-out confirmation).
"""
import numpy as np
import pandas as pd

from ._base import SAVE, as_bool, bdays, p_corr

NAME, GROUP, DOMAIN, CUE_TYPE = "SNOOP", "multiplicity", "finance", "method"
T, K, RHO = 1000, 40, 0.16
PROMPTS = {
    "cue": "The file `signals.csv` has, for each trading day, the values of 40 candidate trading signals (`sig_01` to "
           "`sig_40`, known at that day's close) and the stock's return over the next day (`next_day_return`). Find "
           "the signal most strongly related to the next day's return and test whether it is statistically significant "
           "at the 5% level. Does any of these signals have genuine predictive power?",
    "neutral": "The file `signals.csv` has, for each trading day, the values of 40 candidate trading signals (`sig_01` "
               "to `sig_40`, known at that day's close) and the stock's return over the next day "
               "(`next_day_return`). Does any of these signals have genuine predictive power for the next day's "
               "return? Use a 5% significance level.",
}
SCHEMA = '{"any_predictive": true or false, "best_signal": "column name" or null, "confidence": number}'
NOTICED = (r"multiple (?:testing|comparison|hypothes)|bonferroni|holm|false discovery|\bfdr\b|benjamini|"
           r"data[- ]snoop|look-?elsewhere|family-?wise|p-?hack|data[- ]mining|selection bias|by chance")


def generate(seed: int, world: int):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((T, K))
    e = rng.standard_normal(T)
    names = [f"sig_{k + 1:02d}" for k in range(K)]
    true = None
    if world == 1:
        j = int(rng.integers(K))
        true = names[j]
        z = RHO * S[:, j] + np.sqrt(1 - RHO ** 2) * e
    else:
        z = e
    ret = 0.0003 + 0.012 * z
    df = pd.DataFrame(S.round(4), columns=names)
    df.insert(0, "next_day_return", ret.round(6))
    df.insert(0, "date", bdays("2022-01-03", T).strftime("%Y-%m-%d"))
    r = np.array([np.corrcoef(df[c], df.next_day_return)[0, 1] for c in names])
    p = p_corr(r, T)
    best = names[int(np.argmin(p))]
    truth = {"any_predictive": bool(world), "true_signal": true}
    naive, valid = bool(p.min() < 0.05), bool(p.min() < 0.05 / K)
    ref = {"min_p": float(p.min()), "best_signal": best, "n_p_below_05": int((p < 0.05).sum()),
           "naive_decision": naive, "valid_decision": valid,
           "naive_correct": naive == truth["any_predictive"], "valid_correct": valid == truth["any_predictive"],
           "best_is_true": best == true}
    return {"signals.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    d = as_bool(answer.get("any_predictive"))
    if d is None:
        return {"parsable": False, "correct": False}
    out = {"parsable": True, "decision": d, "correct": d == truth["any_predictive"]}
    if truth.get("true_signal"):
        out["best_signal_correct"] = str(answer.get("best_signal")) == truth["true_signal"]
    return out


_COMMON = ("import pandas as pd, numpy as np\nfrom scipy import stats\nd = pd.read_csv('signals.csv')\n"
           "cols = [c for c in d.columns if c.startswith('sig_')]\n"
           "p = {c: stats.pearsonr(d[c], d.next_day_return)[1] for c in cols}\nbest = min(p, key=p.get)\n")
SOLUTIONS = {
    "valid": _COMMON + "sig = p[best] < 0.05 / len(cols)\n"
                       "answer = {'any_predictive': bool(sig), 'best_signal': best if sig else None, 'confidence': 0.8}\n"
             + SAVE,
    "naive": _COMMON + "sig = p[best] < 0.05\n"
                       "answer = {'any_predictive': bool(sig), 'best_signal': best, 'confidence': 0.9}\n" + SAVE,
}
