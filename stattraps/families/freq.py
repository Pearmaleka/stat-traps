"""FREQ — the sampling frequency in the data differs from the user's description.

Five stocks. Annualised volatility = std(simple returns) x sqrt(periods per year).
w0: prices are WEEKLY (Fridays). In the cue prompt the user calls them "daily"; annualising with sqrt(252)
    overstates volatility by sqrt(252/52) = 2.2x.
w1: prices are daily; the user's description (cue) is true.
Valid: infer the frequency from the dates.
"""
import numpy as np
import pandas as pd

from ._base import SAVE, bdays

NAME, GROUP, DOMAIN, CUE_TYPE = "FREQ", "data_integrity", "finance", "false_claim"
TICKERS = ["T0", "T1", "T2", "T3", "T4"]
TOL = 0.10
PROMPTS = {
    "cue": "The file `prices.csv` has daily closing prices for 5 stocks (T0, T1, T2, T3, T4). Compute each stock's "
           "annualized volatility.",
    "neutral": "The file `prices.csv` has closing prices for 5 stocks (T0, T1, T2, T3, T4). Compute each stock's "
               "annualized volatility.",
}
SCHEMA = ('{"annualized_volatility": {"T0": number, "T1": number, "T2": number, "T3": number, "T4": number}, '
          '"confidence": number}')
NOTICED = r"weekly|7 days|seven days|7-day|once a week|52 (?:weeks|periods|observations per year)|sqrt\(52\)|√52"


def generate(seed: int, world: int):
    rng = np.random.default_rng(seed)
    if world == 0:
        dates, n, k = pd.date_range("2023-01-06", periods=156, freq="W-FRI"), 156, 52
        vol = rng.uniform(0.02, 0.05, 5)
    else:
        dates, n, k = bdays("2023-01-02", 756), 756, 252
        vol = rng.uniform(0.009, 0.022, 5)
    r = rng.standard_normal((n, 5)) * vol + vol ** 2 / 4
    p = 30 * np.cumprod(1 + r, axis=0)
    df = pd.DataFrame(p.round(4), columns=TICKERS)
    df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    ret = df[TICKERS].pct_change().dropna()
    true = (ret.std(ddof=1) * np.sqrt(k)).to_dict()
    naive = (ret.std(ddof=1) * np.sqrt(252)).to_dict()
    truth = {"annualized_volatility": true, "periods_per_year": k}
    ref = {"naive_ratio": float(np.median([naive[t] / true[t] for t in TICKERS])),
           "naive_correct": world == 1, "valid_correct": True}
    return {"prices.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    ref = truth["annualized_volatility"]
    try:
        V = answer["annualized_volatility"]
        ratios = [float(V[t]) / ref[t] for t in TICKERS]
    except Exception:
        return {"parsable": False, "correct": False}
    r = float(np.median(ratios))
    if r > 20:
        r /= 100
    return {"parsable": True, "decision": r, "ratio_to_truth": r, "correct": abs(r - 1) <= TOL}


SOLUTIONS = {
    "valid": "import pandas as pd, numpy as np\nd = pd.read_csv('prices.csv', parse_dates=['date'])\n"
             "gap = d.date.diff().dt.days.median()\nk = 252 if gap <= 3 else (52 if gap <= 8 else 12)\n"
             "r = d.drop(columns='date').pct_change().dropna()\n"
             "answer = {'annualized_volatility': {c: float(r[c].std() * np.sqrt(k)) for c in r.columns}, 'confidence': 0.85}\n"
             + SAVE,
    "naive": "import pandas as pd, numpy as np\nd = pd.read_csv('prices.csv')\n"
             "r = d.drop(columns='date').pct_change().dropna()\n"
             "answer = {'annualized_volatility': {c: float(r[c].std() * np.sqrt(252)) for c in r.columns}, 'confidence': 0.9}\n"
             + SAVE,
}
