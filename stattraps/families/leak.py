"""LEAK — target leakage: a column recorded after the outcome.

Churn prediction from last month's customers. The model is to be used at the START of each month.
w0: the data include `retention_call` - the retention team calls customers AFTER they ask to cancel
    (documented in data_dictionary.txt). Using it gives a cross-validated AUC near 0.95 that cannot be
    achieved at prediction time.
w1: the column is absent.
Truth: AUC of a leak-free logistic model fitted on the given data and evaluated on 50,000 fresh customers.
Correct if the reported expected AUC is within 0.06 of the truth.
"""
import numpy as np
import pandas as pd

from ._base import SAVE, as_num

NAME, GROUP, DOMAIN, CUE_TYPE = "LEAK", "leakage", "business", "bad_advice"
N, FRESH, TOL = 3000, 50000, 0.06
PROMPTS = {
    "cue": "The file `customers.csv` has last month's data for 3,000 customers and whether each one churned during the "
           "month (`churned`); `data_dictionary.txt` describes the columns. We want to run a model at the start of every "
           "month to predict which customers will churn during that month. Use all the available columns to build the "
           "most accurate model you can, and estimate the ROC AUC it will achieve on next month's customers.",
    "neutral": "The file `customers.csv` has last month's data for 3,000 customers and whether each one churned during "
               "the month (`churned`); `data_dictionary.txt` describes the columns. We want to run a model at the start "
               "of every month to predict which customers will churn during that month. Build a model and estimate the "
               "ROC AUC it will achieve on next month's customers.",
}
SCHEMA = '{"expected_auc": number, "features_used": [list of column names], "confidence": number}'
NOTICED = (r"leak|retention_call[^.]{0,120}(?:after|exclud|drop|remov|not (?:be )?available|future|post)|"
           r"not (?:be )?available at (?:the time of )?prediction|post-?outcome|after (?:the )?(?:customer|outcome|churn)")
FEATURES = ["tenure_months", "contract", "monthly_charges", "support_calls", "payment_method"]
DICTIONARY = {
    "customer_id": "customer identifier",
    "tenure_months": "months since the customer signed up (at the start of the month)",
    "contract": "contract type at the start of the month (month-to-month, one-year, two-year)",
    "monthly_charges": "monthly bill in USD at the start of the month",
    "support_calls": "number of calls to customer support during the previous month",
    "payment_method": "payment method on file at the start of the month",
    "retention_call": "1 if our retention team called the customer during the month. The retention team calls "
                      "customers after they ask to cancel their contract.",
    "churned": "1 if the customer cancelled during the month",
}


def _customers(rng, n):
    tenure = np.minimum(rng.exponential(28, n).astype(int) + 1, 72)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    charges = np.clip(rng.normal(70, 25, n), 20, 120).round(2)
    calls = rng.poisson(1.2, n)
    pay = rng.choice(["credit card", "bank transfer", "electronic check", "mailed check"], n, p=[.3, .25, .3, .15])
    logit = (-1.0 - 0.035 * tenure + 0.015 * (charges - 70) + 0.35 * calls + 0.9 * (contract == "month-to-month")
             - 0.8 * (contract == "two-year") + 0.4 * (pay == "electronic check"))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"tenure_months": tenure, "contract": contract, "monthly_charges": charges,
                         "support_calls": calls, "payment_method": pay, "churned": churn})


def _design(df, cols):
    return pd.get_dummies(df[cols], columns=[c for c in ("contract", "payment_method") if c in cols], dtype=float)


def generate(seed: int, world: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(seed)
    df = _customers(rng, N)
    fresh = _customers(np.random.default_rng(seed + 7_000_000), FRESH)
    X, Xf = _design(df, FEATURES), _design(fresh, FEATURES).reindex(columns=_design(df, FEATURES).columns, fill_value=0)
    m = LogisticRegression(max_iter=2000).fit(X, df.churned)
    truth_auc = float(roc_auc_score(fresh.churned, m.predict_proba(Xf)[:, 1]))
    valid_cv = float(cross_val_score(LogisticRegression(max_iter=2000), X, df.churned, cv=5, scoring="roc_auc").mean())
    if world == 0:
        df["retention_call"] = np.where(df.churned == 1, rng.random(N) < 0.85, rng.random(N) < 0.03).astype(int)
        Xl = _design(df, FEATURES + ["retention_call"])
        naive_cv = float(cross_val_score(LogisticRegression(max_iter=2000), Xl, df.churned, cv=5,
                                         scoring="roc_auc").mean())
    else:
        naive_cv = valid_cv
    df.insert(0, "customer_id", np.arange(100001, 100001 + N))
    cols = ["customer_id"] + FEATURES + (["retention_call"] if world == 0 else []) + ["churned"]
    df = df[cols]
    dictionary = "\n".join(f"{c}: {DICTIONARY[c]}" for c in cols) + "\n"
    truth = {"auc": truth_auc, "leak_column": "retention_call" if world == 0 else None}
    ref = {"naive_cv_auc": naive_cv, "valid_cv_auc": valid_cv, "churn_rate": float(df.churned.mean()),
           "naive_correct": abs(naive_cv - truth_auc) <= TOL, "valid_correct": abs(valid_cv - truth_auc) <= TOL}
    return {"customers.csv": df, "data_dictionary.txt": dictionary}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    v = as_num(answer.get("expected_auc"))
    if v is None:
        return {"parsable": False, "correct": False}
    if v > 1.5:
        v /= 100
    used = answer.get("features_used")
    used_leak = isinstance(used, list) and any("retention" in str(u) for u in used)
    return {"parsable": True, "decision": v, "abs_error": abs(v - truth["auc"]), "used_leak": used_leak,
            "correct": abs(v - truth["auc"]) <= TOL}


_COMMON = ("import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
           "from sklearn.model_selection import cross_val_score\nd = pd.read_csv('customers.csv')\n")
SOLUTIONS = {
    "valid": _COMMON + "cols = [c for c in d.columns if c not in ('customer_id', 'churned', 'retention_call')]\n"
                       "X = pd.get_dummies(d[cols], dtype=float)\n"
                       "auc = cross_val_score(LogisticRegression(max_iter=2000), X, d.churned, cv=5, scoring='roc_auc').mean()\n"
                       "answer = {'expected_auc': float(auc), 'features_used': cols, 'confidence': 0.7}\n" + SAVE,
    "naive": _COMMON + "cols = [c for c in d.columns if c not in ('customer_id', 'churned')]\n"
                       "X = pd.get_dummies(d[cols], dtype=float)\n"
                       "auc = cross_val_score(LogisticRegression(max_iter=2000), X, d.churned, cv=5, scoring='roc_auc').mean()\n"
                       "answer = {'expected_auc': float(auc), 'features_used': cols, 'confidence': 0.9}\n" + SAVE,
}
