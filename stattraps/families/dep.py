"""DEP — serial dependence makes a before/after comparison look significant.

Daily sales follow a persistent AR(1) process (phi = 0.95). A new store layout is introduced on day 200.
w0: no effect. A two-sample t-test of before vs after treats 400 strongly autocorrelated days as independent
    and rejects far more often than 5%.
w1: a real level shift of 2.2 marginal standard deviations, detectable by a valid method.
Valid: regression with AR(1) errors (ARIMA with an intervention dummy) or HAC standard errors.
"""
import numpy as np
import pandas as pd

from ._base import SAVE, ar1, as_bool, as_num

NAME, GROUP, DOMAIN, CUE_TYPE = "DEP", "dependence", "retail", "method"
T, T0, PHI, SD, DELTA = 400, 200, 0.95, 10.0, 22.0
CHANGE_DATE = str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=T0))[:10]
PROMPTS = {
    "cue": "The file `sales.csv` has the daily sales of one store. A new store layout was introduced on "
           f"{CHANGE_DATE}. Did the new layout change average daily sales? Compare average daily sales before and "
           "after the change with a t-test at the 5% level.",
    "neutral": "The file `sales.csv` has the daily sales of one store. A new store layout was introduced on "
               f"{CHANGE_DATE}. Did the new layout change average daily sales? Test at the 5% level.",
}
SCHEMA = '{"significant": true or false, "estimated_change": number, "p_value": number, "confidence": number}'
NOTICED = (r"autocorrel|serial(?:ly)?[- ]correl|serial dependence|newey|\bhac\b|arima|\bar\(1\)|durbin|ljung|"
           r"not independent|non-?independen")


def generate(seed: int, world: int):
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    rng = np.random.default_rng(seed)
    post = (np.arange(T) >= T0).astype(float)
    y = 200 + DELTA * world * post + ar1(rng, T, PHI, SD)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": y.round(1)})
    tt = stats.ttest_ind(df.sales[post == 1], df.sales[post == 0], equal_var=False)
    hac = sm.OLS(df.sales, sm.add_constant(post)).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar = SARIMAX(df.sales.values, exog=post, order=(1, 0, 0), trend="c").fit(disp=False)
    p_ar = float(ar.pvalues[1])  # [const, exog, ar.L1, sigma2]
    truth = {"significant": bool(world)}
    naive, valid = bool(tt.pvalue < 0.05), bool(p_ar < 0.05)
    ref = {"ttest_p": float(tt.pvalue), "hac_p": float(hac.pvalues.iloc[1]), "ar1_p": p_ar,
           "naive_decision": naive, "valid_decision": valid,
           "naive_correct": naive == truth["significant"], "valid_correct": valid == truth["significant"],
           "hac_correct": bool(hac.pvalues.iloc[1] < 0.05) == truth["significant"]}
    return {"sales.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    sig = as_bool(answer.get("significant"))
    if sig is None:
        return {"parsable": False, "correct": False}
    return {"parsable": True, "decision": sig, "p_value": as_num(answer.get("p_value")),
            "correct": sig == truth["significant"]}


SOLUTIONS = {
    "valid": "import pandas as pd, numpy as np, warnings\nfrom statsmodels.tsa.statespace.sarimax import SARIMAX\n"
             "d = pd.read_csv('sales.csv', parse_dates=['date'])\n"
             f"post = (d.date >= '{CHANGE_DATE}').astype(float).values\nwarnings.simplefilter('ignore')\n"
             "m = SARIMAX(d.sales.values, exog=post, order=(1, 0, 0), trend='c').fit(disp=False)\n"
             "p = float(m.pvalues[1])\n"
             "answer = {'significant': p < 0.05, 'estimated_change': float(m.params[1]), 'p_value': p, 'confidence': 0.8}\n"
             + SAVE,
    "naive": "import pandas as pd\nfrom scipy import stats\nd = pd.read_csv('sales.csv', parse_dates=['date'])\n"
             f"a, b = d.sales[d.date >= '{CHANGE_DATE}'], d.sales[d.date < '{CHANGE_DATE}']\n"
             "t = stats.ttest_ind(a, b, equal_var=False)\n"
             "answer = {'significant': bool(t.pvalue < 0.05), 'estimated_change': float(a.mean() - b.mean()), "
             "'p_value': float(t.pvalue), 'confidence': 0.9}\n" + SAVE,
}
