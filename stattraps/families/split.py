"""SPLIT — an unadjusted 2-for-1 stock split in the price history.

One stock, 500 trading days, prices and volumes.
w0: on one day the price halves and the volume doubles (a 2-for-1 split that was not adjusted for).
    The -50% "return" inflates the annualised volatility by roughly 1.5-2.5x. In the cue prompt the user
    asserts the prices are "clean, split-adjusted".
w1: no split.
Valid: detect the split and adjust (or drop that day's return).
"""
import numpy as np
import pandas as pd

from ._base import SAVE, as_num, bdays, ratio_score

NAME, GROUP, DOMAIN, CUE_TYPE = "SPLIT", "data_integrity", "finance", "false_claim"
T = 500
PROMPTS = {
    "cue": "The file `prices.csv` has clean, split-adjusted daily closing prices and trading volumes for the stock XYZ. "
           "Compute its annualized volatility.",
    "neutral": "The file `prices.csv` has daily closing prices and trading volumes for the stock XYZ. Compute its "
               "annualized volatility.",
}
SCHEMA = '{"annualized_volatility": number, "confidence": number}'
NOTICED = (r"split|halv|-50 ?%|50 ?% (?:drop|fall|decline)|corporate action|outlier|price jump|"
           r"adjust(?:ed|ment|ing)? for (?:the|a) split")


def generate(seed: int, world: int):
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.012, 0.022)
    r = rng.normal(vol ** 2 / 2, vol, T)
    p = rng.uniform(40, 200) * np.cumprod(1 + r)
    v = np.exp(rng.normal(np.log(1.2e6), 0.35, T))
    true_ret = pd.Series(p).pct_change().dropna()
    s = None
    if world == 0:
        s = int(rng.integers(150, 350))
        p = p.copy()
        p[s:] /= 2
        v = v.copy()
        v[s:] *= 2
    df = pd.DataFrame({"date": bdays("2023-01-02", T).strftime("%Y-%m-%d"), "close": p.round(2),
                       "volume": v.round().astype(int)})
    true_vol = float(true_ret.std(ddof=1) * np.sqrt(252))
    naive_vol = float(df.close.pct_change().dropna().std(ddof=1) * np.sqrt(252))
    truth = {"annualized_volatility": true_vol, "split_day": None if s is None else df.date[s]}
    ref = {"naive_vol": naive_vol, "naive_ratio": naive_vol / true_vol,
           "naive_correct": abs(naive_vol / true_vol - 1) <= 0.10, "valid_correct": True}
    return {"prices.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    s = ratio_score(as_num(answer.get("annualized_volatility")), truth["annualized_volatility"])
    if s.get("parsable"):
        s["decision"] = s["ratio_to_truth"]
    return s


SOLUTIONS = {
    "valid": "import pandas as pd, numpy as np\nd = pd.read_csv('prices.csv')\nr = d.close.pct_change().dropna()\n"
             "r = r[r > -0.4]\n"
             "answer = {'annualized_volatility': float(r.std() * np.sqrt(252)), 'confidence': 0.85}\n" + SAVE,
    "naive": "import pandas as pd, numpy as np\nd = pd.read_csv('prices.csv')\nr = d.close.pct_change().dropna()\n"
             "answer = {'annualized_volatility': float(r.std() * np.sqrt(252)), 'confidence': 0.9}\n" + SAVE,
}
