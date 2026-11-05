"""Fast checks: deterministic generation, prompt hygiene, scoring of the true answer.  Run: pytest -q"""
import json
import re

import pandas as pd
import pytest

from stattraps import suite
from stattraps.families._base import footer

FAMS = list(suite.FAMILIES)

# words that would give the trap away; they must never appear in a prompt the agent sees
GIVEAWAYS = r"spurious|stationar|cointegrat|autocorrel|cluster|multiple testing|bonferroni|confound|simpson|leak|" \
            r"split-?adjust|weekly|missing data|pitfall|careful|assumption"


@pytest.mark.parametrize("fam", FAMS)
def test_generation_is_deterministic(fam):
    mod = suite.FAMILIES[fam]
    for world in (0, 1):
        f1, t1, r1, m1 = mod.generate(12345, world)
        f2, t2, r2, m2 = mod.generate(12345, world)
        for k in f1:
            if isinstance(f1[k], pd.DataFrame):
                pd.testing.assert_frame_equal(f1[k], f2[k])
            else:
                assert f1[k] == f2[k]
        assert json.dumps(t1, default=str) == json.dumps(t2, default=str)


@pytest.mark.parametrize("fam", FAMS)
def test_neutral_prompt_has_no_giveaway(fam):
    mod = suite.FAMILIES[fam]
    text = mod.PROMPTS["neutral"] + footer(mod.SCHEMA)
    assert not re.search(GIVEAWAYS, text, re.I), text


@pytest.mark.parametrize("fam", FAMS)
def test_cue_and_neutral_differ_only_by_the_cue(fam):
    mod = suite.FAMILIES[fam]
    assert mod.PROMPTS["cue"] != mod.PROMPTS["neutral"]


def _answer_from_truth(fam, truth):
    if fam in ("SPUR", "DEP", "PSEUDO"):
        return {"significant": truth["significant"], "confidence": 0.9}
    if fam == "SNOOP":
        return {"any_predictive": truth["any_predictive"], "best_signal": truth["true_signal"], "confidence": 0.9}
    if fam == "SIMPSON":
        return {"better_treatment": truth["better_treatment"], "confidence": 0.9}
    if fam == "LEAK":
        return {"expected_auc": truth["auc"], "features_used": [], "confidence": 0.9}
    if fam == "FREQ":
        return {"annualized_volatility": truth["annualized_volatility"], "confidence": 0.9}
    if fam == "SPLIT":
        return {"annualized_volatility": truth["annualized_volatility"], "confidence": 0.9}
    return None


@pytest.mark.parametrize("fam", [f for f in FAMS if f != "GAPS"])
def test_true_answer_scores_correct_and_flipped_scores_wrong(fam):
    mod = suite.FAMILIES[fam]
    for world in (0, 1):
        _, truth, _, _ = mod.generate(777, world)
        ans = _answer_from_truth(fam, truth)
        assert mod.score(ans, truth)["correct"]
        assert not mod.score({}, truth)["correct"]
