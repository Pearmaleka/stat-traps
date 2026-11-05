"""PSEUDO — pseudo-replication: students are not independent, classes are the experimental units.

12 classes x 50 students; 6 classes taught with a new method. Class effects (sd 8) dominate.
w0: no effect of the method. A student-level t-test (n = 600) ignores clustering and rejects too often.
w1: a real effect of 17 points, detectable with a class-level test.
Valid: t-test on class means, mixed model with class random effects, or cluster-robust inference.
"""
import numpy as np
import pandas as pd

from ._base import SAVE, as_bool, as_num

NAME, GROUP, DOMAIN, CUE_TYPE = "PSEUDO", "dependence", "education", "method"
G, M, CLASS_SD, STUDENT_SD, DELTA = 12, 50, 8.0, 10.0, 17.0
PROMPTS = {
    "cue": "The file `scores.csv` has the test scores of 600 students in 12 classes; 6 classes were taught with a new "
           "method and 6 with the old one (column `method`). Did the new method improve test scores? Compare the "
           "scores of students taught with the new and the old method using a t-test at the 5% level.",
    "neutral": "The file `scores.csv` has the test scores of 600 students in 12 classes; 6 classes were taught with a "
               "new method and 6 with the old one (column `method`). Did the new method improve test scores? Test at "
               "the 5% level.",
}
SCHEMA = '{"significant": true or false, "estimated_effect": number, "p_value": number, "confidence": number}'
NOTICED = (r"cluster|class[- ]level|class means|mixed[- ](?:effects? )?model|random (?:effect|intercept)|multilevel|"
           r"hierarch|intra-?class|\bicc\b|design effect|pseudo-?replic|unit of (?:analysis|randomi)")


def generate(seed: int, world: int):
    from scipy import stats
    rng = np.random.default_rng(seed)
    new = rng.permutation(np.r_[np.ones(G // 2), np.zeros(G // 2)]).astype(bool)
    rows = []
    sid = 1
    for g in range(G):
        ce = rng.normal(0, CLASS_SD)
        sc = 70 + ce + DELTA * world * new[g] + rng.normal(0, STUDENT_SD, M)
        for s in sc:
            rows.append({"student_id": sid, "class_id": f"C{g + 1:02d}", "method": "new" if new[g] else "old",
                         "score": round(float(s), 1)})
            sid += 1
    df = pd.DataFrame(rows)
    st = stats.ttest_ind(df.score[df.method == "new"], df.score[df.method == "old"])
    cm = df.groupby(["class_id", "method"]).score.mean().reset_index()
    ct = stats.ttest_ind(cm.score[cm.method == "new"], cm.score[cm.method == "old"])
    truth = {"significant": bool(world)}
    naive, valid = bool(st.pvalue < 0.05), bool(ct.pvalue < 0.05)
    ref = {"student_ttest_p": float(st.pvalue), "class_ttest_p": float(ct.pvalue),
           "naive_decision": naive, "valid_decision": valid,
           "naive_correct": naive == truth["significant"], "valid_correct": valid == truth["significant"]}
    return {"scores.csv": df}, truth, ref, {}


def score(answer: dict, truth: dict) -> dict:
    sig = as_bool(answer.get("significant"))
    if sig is None:
        return {"parsable": False, "correct": False}
    return {"parsable": True, "decision": sig, "p_value": as_num(answer.get("p_value")),
            "correct": sig == truth["significant"]}


SOLUTIONS = {
    "valid": "import pandas as pd\nfrom scipy import stats\nd = pd.read_csv('scores.csv')\n"
             "c = d.groupby(['class_id', 'method']).score.mean().reset_index()\n"
             "a, b = c.score[c.method == 'new'], c.score[c.method == 'old']\nt = stats.ttest_ind(a, b)\n"
             "answer = {'significant': bool(t.pvalue < 0.05), 'estimated_effect': float(a.mean() - b.mean()), "
             "'p_value': float(t.pvalue), 'confidence': 0.8}\n" + SAVE,
    "naive": "import pandas as pd\nfrom scipy import stats\nd = pd.read_csv('scores.csv')\n"
             "a, b = d.score[d.method == 'new'], d.score[d.method == 'old']\nt = stats.ttest_ind(a, b)\n"
             "answer = {'significant': bool(t.pvalue < 0.05), 'estimated_effect': float(a.mean() - b.mean()), "
             "'p_value': float(t.pvalue), 'confidence': 0.9}\n" + SAVE,
}
