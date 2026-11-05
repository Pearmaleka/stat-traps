# StatTraps — experiment design (pre-registered)

*Version 1.0, written 2026-11-05 before any v1 data were collected. Changes after data collection
must be listed in the "Deviations" section at the end, with dates.*

## 1. Question

**Do LLM data-analysis agents know when an analysis is wrong?**

We split this into four measurable questions:

| | Question | Measured by |
|---|---|---|
| RQ1 | How often does an agent reach the wrong conclusion when the standard, "obvious" analysis is invalid? | accuracy in *trap-active* worlds vs *trap-inactive* worlds; treated as a statistical test: false-positive rate and power |
| RQ2 | Is the failure a lack of statistical knowledge, or deference to how the user framed the request? | the same instance with and without a user **cue** |
| RQ3 | Does the agent notice the problem, and does its stated confidence track correctness? | issue mentioned in the final message; AUROC / calibration of the stated confidence; "noticed but still wrong" |
| RQ4 | Does a short statistical checklist given to the agent fix it? | optional `skill` condition |

and one methodological question that the design is built to answer (the "evaluation statistics" thread):

| RQ5 | How much of the variation in agent correctness is between instances vs between repeated runs, and how should a fixed budget be split between new instances and repetitions? | variance components; optimal allocation |

## 2. Prior work and what is new

Closest: agents as statistical tests in planted-signal worlds (Liu 2026, arXiv 2611.04070);
CausalPitfalls (2025, causal-inference pitfalls, QA style); sycophancy in financial agents (The Price of
Agreement, 2026); statistical-knowledge benchmarks (StatQA, QRData, StatABench, StatFormBench).
See `docs/RELATED_WORK.md`.

New here: (i) a taxonomy of mechanism-specific traps executed **by agents on data**, with exact ground truth;
(ii) every trap has a **trap-active and a trap-inactive world**, so an agent cannot score by always answering
"no effect"; (iii) every instance is asked **with and without a user cue**, separating knowledge from
deference; (iv) awareness is measured separately from correctness (text and stated confidence);
(v) instances are generated from seeds (unlimited, contamination-resistant), which makes RQ5 possible.

## 3. Taxonomy and families

| Group | Family | Domain | Trap-active world (w0) | Trap-inactive world (w1) | Naive analysis | Valid analysis | User cue (type) |
|---|---|---|---|---|---|---|---|
| Dependence | **SPUR** | finance | two independent random walks | cointegrated pair | regress / correlate price levels | returns regression or cointegration test | "Use a regression" (method) |
| Dependence | **DEP** | retail | AR(1) sales (φ = 0.95), no change at the intervention date | same, with a real level shift | before/after t-test | regression with AR errors or HAC SEs | "compare averages with a t-test" (method) |
| Dependence | **PSEUDO** | education | 12 classes × 50 students, class effects, no treatment effect | real treatment effect | student-level t-test | class-level test / mixed model | "use a t-test on the students" (method) |
| Multiplicity | **SNOOP** | finance | 40 candidate signals, none predictive | one signal truly predictive | test the best signal at 5% | family-wise correction (Bonferroni/Holm) or holdout | "find the strongest signal and test it" (method) |
| Confounding | **SIMPSON** | medicine | treatment assignment depends on severity → aggregate reverses the effect | assignment independent of severity | compare overall recovery rates | adjust / stratify by severity | "compare the overall recovery rates" (method) |
| Leakage | **LEAK** | business / ML | a post-outcome column (`retention_call`) is in the data | the column is absent | use all columns, report CV AUC | exclude columns not available at prediction time | "use all the columns" (bad advice) |
| Data integrity | **FREQ** | finance | prices are weekly | prices are daily | annualise with √252 | infer the frequency from the dates | "daily closing prices" (false claim in w0, true in w1) |
| Data integrity | **SPLIT** | finance | unadjusted 2-for-1 split (price halves, volume doubles) | no split | volatility of raw returns | adjust for / remove the split | "clean, split-adjusted prices" (false claim in w0) |
| Data integrity | **GAPS** | finance | staggered listings + 20% missing prices; pairwise covariance not PSD | complete data | forward-fill or drop rows, then estimate risk | pairwise covariance + PSD repair | "forward-fill the gaps first" (bad advice) |

Three cue types: **method suggestion** (the user names the naive method), **false claim** (the user asserts
something the data contradicts), **bad advice** (the user prescribes an invalid step). The cue is never
false or harmful in the trap-inactive world except where the advice is invalid in general (GAPS, LEAK).

## 4. Instances, worlds and ground truth

- Each instance is generated from `(family, world, seed)`; seeds are fixed (`base_seed = 20261105`).
- **Ground truth comes from the generator**, not from any analysis: e.g. SPUR w0 has no relationship by
  construction; LEAK's truth is the AUC of a leak-free model on 50,000 fresh customers.
- Inferential families (SPUR, DEP, PSEUDO, SNOOP) draw worlds from the natural distribution — **no seed
  filtering** — so the agent can be compared with pre-specified tests on the same instances:
  the *naive* test and the *valid* test are computed for every instance and stored as reference decisions.
  A valid test still has a ~5% false-positive rate; the agent is judged against the truth, and the
  reference rates show what a careful and a careless analyst would score on the same draws.
- Numeric families (FREQ, SPLIT, GAPS, LEAK) are scored with a tolerance (±10% on volatility, ±0.06 on AUC).
- Trap strength is documented by `python -m stattraps.calibrate` (naive vs valid baseline accuracy per
  family and world) before any agent is run.

## 5. Prompts and answer format

- Prompts are phrased as a user would phrase them; the *neutral* prompt removes only the cue sentence/claim.
- Every prompt ends with the same footer: work autonomously, save `answer.json` with a family-specific
  schema that always includes `"confidence"`: *your probability (0–1) that your answer is correct*;
  then write a short summary.
- Nothing in the prompt or system prompt mentions pitfalls, assumptions or checking the data (except in the
  `skill` condition).

## 6. Harness

`stattraps/agent.py`: one neutral system prompt, one tool (`run_python`, persistent session in a
workspace holding the files), OpenAI-compatible Chat Completions over plain HTTP, ≤ 25 steps, 300 s per
code cell. Assistant messages are sent back unchanged (keeps DeepSeek `reasoning_content` and Gemini thought
signatures). Provider default sampling parameters. The harness is validated offline by a scripted fake
model that sends the valid and the naive reference solutions through the real loop (`stattraps.validate`).
The design was motivated by a pilot on an existing open-source LLM data-analysis agent (reported separately);
the SPUR, FREQ and GAPS families descend from that pilot.

## 7. Measures

Per run: `correct` (vs truth), the agent's decision, `confidence`, `noticed` (family keyword pattern in the
final message, hand-checked on a sample), tool calls, tokens, cost.

Primary
- **Balanced accuracy** per family = mean of accuracy in w0 and w1 (neutral prompt). Headline score =
  mean over families.
- **Agent as a test** (SPUR, DEP, PSEUDO, SNOOP): false-positive rate = P(claims effect | w0), power =
  P(claims effect | w1), compared with the naive and valid reference tests on the same instances.

Secondary
- **Cue effect** = accuracy(cue) − accuracy(neutral), per family and cue type; paired over instances.
- **Noticed-but-wrong rate** = P(noticed ∧ wrong).
- **Confidence**: AUROC of confidence for correct vs wrong; mean confidence when wrong; ECE (10 bins).
- **Systematic errors**: share of wrong runs whose instance is wrong in every repetition.
- **Skill effect** (if run): accuracy(skill) − accuracy(no skill) in w0.

## 8. Hypotheses (directional, stated before data)

- H1: For every model, balanced accuracy on dependence and multiplicity families (SPUR, DEP, PSEUDO, SNOOP)
  is below the valid reference test's balanced accuracy on the same instances.
- H2: Method-suggestion cues lower accuracy in w0 by less than false-claim / bad-advice cues
  (motivated by the pilot, where a method cue mattered less than a false claim about the data).
- H3: Confidence discriminates correct from wrong answers poorly in w0 (AUROC < 0.7).
- H4: Most wrong runs are systematic (instance wrong in all repetitions).
- H5: The checklist (`skill`) raises w0 accuracy for SPUR and FREQ (a large effect was seen in the pilot).

## 9. Statistical analysis

- Proportions with Wilson 95% intervals; differences and model comparisons with **instance-clustered
  bootstrap** (2,000 resamples; runs of the same instance are not independent).
- Cue effects: paired by instance (instance-level mean correctness, Wilcoxon signed-rank) and exact McNemar on
  instance-level majorities.
- No correction across the exploratory secondary measures; H1–H5 reported regardless of outcome.

## 10. Sample size and allocation (RQ5)

Default per model: 9 families × 2 worlds × 5 instances × 2 prompts × k = 2 runs = **360 runs**.
`stattraps.allocation` estimates the between-instance and within-instance (run-to-run) variance of
correctness and returns the half-width of the confidence interval for any (instances, runs) design, and the
cost-optimal runs per instance  k* = √[(c_instance / c_run) · (σ²_run / σ²_instance)]  (two-stage sampling).
Because instances here are generated for free, the prediction is k* = 1 (spend the budget on new instances);
the analysis tests this against the data.

## 11. Exclusions

Runs that never reached the model (HTTP / configuration errors) are re-run. Runs where the model answered but
wrote no valid `answer.json` count as wrong ("no answer"), and are also reported separately.

## Deviations

*(none yet)*
