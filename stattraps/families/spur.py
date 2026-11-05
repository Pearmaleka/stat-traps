"""SPUR — spurious regression between non-stationary price series.

w0: two independent random walks. Regressing one price level on the other gives tiny p-values (Granger &
    Newbold 1974); returns or a cointegration test correctly find nothing.
w1: a cointegrated pair (B tracks A plus a stationary spread): a real relationship, found by any method.
"""
import numpy as np
import pandas as pd

from ._base import SAVE, ar1, as_bool, as_num, bdays

NAME, GROUP, DOMAIN, CUE_TYPE = "SPUR", "dependence", "finance", "method"
PROMPTS = {
    "cue": "The file `prices.csv` has daily closing prices for two stocks, AAA and BBB. Is there a statistically "
           "significant relationship between them? Use a regression and test at the 5% level.",
    "neutral": "The file `prices.csv` has daily closing prices for two stocks, AAA and BBB. Is there a statistically "
               "significant relationship between them? Test at the 5% level.",
}
SCHEMA = '{"significant": true or false, "p_value": number, "confidence": number}'
NOTICED = r"spurious|non-?stationar|unit root|cointegrat|random walk|dickey|\badf\b|durbin|autocorrel"


def generate(seed: int, world: int):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    rng = np.random.default_rng(seed)
    T = 750
    la = np.log(40) + np.cumsum(rng.normal(rng.uniform(-3e-4, 8e-4), 0.015, T))
    if world == 0:
        lb = np.log(60) + np.cumsum(rng.normal(rng.uniform(-3e-4, 8e-4), 0.017, T))
    else:
        beta = rng.uniform(0.7, 1.3)
        lb = np.log(60) + beta * (la - la[0]) + ar1(rng, T, 0.8, 0.02)
    a, b = np.exp(la), np.exp(lb)
    df = pd.DataFrame({"date": bdays("2023-01-02", T).strftime("%Y-%m-%d"), "AAA": a.round(4), "BBB": b.round(4)})
    lv = sm.OLS(df.BBB, sm.add_constant(df.AAA)).fit()
    r = df[["AAA", "BBB"]].pct_change().dropna()
    rt = sm.OLS(r.BBB, sm.add_constant(r.AAA)).fit()
    eg = float(coint(df.BBB, df.AAA)[1])
    truth = {"significant": bool(world)}
    naive, valid = bool(lv.pvalues.iloc[1] < 0.05), bool(rt.pvalues.iloc[1] < 0.05)
    ref = {"levels_p": float(lv.pvalues.iloc[1]), "levels_r2": float(lv.rsquared), "returns_p": float(rt.pvalues.iloc[1]),
           "coint_p": eg, "naive_decision": naive, "valid_decision": valid,
           "naive_correct": naive == truth["significant"], "valid_correct": valid == truth["significant"]}
    return {"prices.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    sig = as_bool(answer.get("significant"))
    if sig is None:
        return {"parsable": False, "correct": False}
    return {"parsable": True, "decision": sig, "p_value": as_num(answer.get("p_value")),
            "correct": sig == truth["significant"]}


SOLUTIONS = {
    "valid": "import pandas as pd, statsmodels.api as sm\nd = pd.read_csv('prices.csv')\n"
             "r = d[['AAA', 'BBB']].pct_change().dropna()\nf = sm.OLS(r.BBB, sm.add_constant(r.AAA)).fit()\n"
             "p = float(f.pvalues.iloc[1])\nanswer = {'significant': p < 0.05, 'p_value': p, 'confidence': 0.8}\n" + SAVE,
    "naive": "import pandas as pd, statsmodels.api as sm\nd = pd.read_csv('prices.csv')\n"
             "f = sm.OLS(d.BBB, sm.add_constant(d.AAA)).fit()\n"
             "p = float(f.pvalues.iloc[1])\nanswer = {'significant': p < 0.05, 'p_value': p, 'confidence': 0.95}\n" + SAVE,
}
