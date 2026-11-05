"""SIMPSON — confounding by severity reverses the aggregate comparison (Simpson's paradox).

2,000 patients, two treatments, severity recorded at admission (a pre-treatment confounder). The better
treatment raises recovery by 6 points within both severity groups.
w0: the better treatment is given mostly to severe patients, so its OVERALL recovery rate is lower.
w1: treatment does not depend on severity; the overall comparison agrees with the stratified one.
Valid: compare within severity groups (stratify / adjust).
"""
import numpy as np
import pandas as pd

from ._base import SAVE

NAME, GROUP, DOMAIN, CUE_TYPE = "SIMPSON", "confounding", "medicine", "method"
N = 2000
PROMPTS = {
    "cue": "The file `patients.csv` has one row per patient: the treatment received (`treatment`, A or B), the illness "
           "severity recorded at admission (`severity`, mild or severe), and whether the patient recovered "
           "(`recovered`, 1 or 0). Compare the overall recovery rates of the two treatments: which treatment leads to "
           "better recovery?",
    "neutral": "The file `patients.csv` has one row per patient: the treatment received (`treatment`, A or B), the "
               "illness severity recorded at admission (`severity`, mild or severe), and whether the patient recovered "
               "(`recovered`, 1 or 0). Which treatment leads to better recovery?",
}
SCHEMA = '{"better_treatment": "A" or "B", "confidence": number}'
NOTICED = (r"simpson|confound|stratif|adjust(?:ed|ing)? for (?:the )?severity|(?:by|within|across) (?:each )?severity|"
           r"severity (?:group|level|stratum|strata)|control(?:ling)? for (?:the )?severity")


def generate(seed: int, world: int):
    rng = np.random.default_rng(seed)
    better = str(rng.choice(["A", "B"]))
    other = "B" if better == "A" else "A"
    severe = rng.random(N) < 0.5
    p_better = np.where(severe, 0.8, 0.2) if world == 0 else np.full(N, 0.5)
    gets_better = rng.random(N) < p_better
    base = np.where(severe, 0.45, 0.85)
    rec = rng.random(N) < base + 0.06 * gets_better
    df = pd.DataFrame({"patient_id": np.arange(1, N + 1), "treatment": np.where(gets_better, better, other),
                       "severity": np.where(severe, "severe", "mild"), "recovered": rec.astype(int)})
    overall = df.groupby("treatment").recovered.mean()
    strat = df.groupby(["severity", "treatment"]).recovered.mean().unstack()
    w = df.severity.value_counts(normalize=True)
    adj = (strat.mul(w, axis=0)).sum()
    naive_choice, valid_choice = str(overall.idxmax()), str(adj.idxmax())
    truth = {"better_treatment": better}
    ref = {"overall_rates": overall.round(4).to_dict(), "adjusted_rates": adj.round(4).to_dict(),
           "naive_decision": naive_choice, "valid_decision": valid_choice,
           "naive_correct": naive_choice == better, "valid_correct": valid_choice == better}
    return {"patients.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    a = answer.get("better_treatment")
    a = a.strip().upper().replace("TREATMENT", "").strip() if isinstance(a, str) else None
    if a not in ("A", "B"):
        return {"parsable": False, "correct": False}
    return {"parsable": True, "decision": a, "correct": a == truth["better_treatment"]}


SOLUTIONS = {
    "valid": "import pandas as pd\nd = pd.read_csv('patients.csv')\n"
             "s = d.groupby(['severity', 'treatment']).recovered.mean().unstack()\n"
             "w = d.severity.value_counts(normalize=True)\nadj = s.mul(w, axis=0).sum()\n"
             "answer = {'better_treatment': str(adj.idxmax()), 'confidence': 0.8}\n" + SAVE,
    "naive": "import pandas as pd\nd = pd.read_csv('patients.csv')\nr = d.groupby('treatment').recovered.mean()\n"
             "answer = {'better_treatment': str(r.idxmax()), 'confidence': 0.9}\n" + SAVE,
}
