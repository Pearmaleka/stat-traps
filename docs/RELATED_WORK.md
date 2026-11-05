# Related work

Literature search of November 2026 (web search; not exhaustive).

## Closest

- **Liu (2026), "Are LLM Research Agents Reliable Judges of Predictability? Evidence from Planted-Signal Worlds"**,
  arXiv:2611.04070. Treats an LLM research agent as a statistical test in simulated worlds with and without a
  planted predictive signal. Agents claim predictability too often in null worlds, state miscalibrated
  probabilities, and shift verdicts when told about prior research. *Difference:* one question type
  (forecast vs benchmark); we use mechanism-specific traps across a taxonomy, cue-controlled prompts, and
  awareness measures. We adopt their "agent as a test" framing for the inferential families.
- **CausalPitfalls (2025)**, "Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in
  Causal Inference", arXiv:2505.13770. 75 questions on causal pitfalls (Simpson's paradox, selection bias…),
  direct and code-assisted prompting; performance drops as hints are removed. *Difference:* causal inference
  only, question-answer format; our SIMPSON family overlaps with theirs.
- **The Price of Agreement (2026)**, "Measuring LLM Sycophancy in Agentic Financial Applications",
  arXiv:2604.24668. Injected user preferences collapse accuracy; most errors are made without acknowledging the
  bias. *Difference:* document QA; our false-claim and bad-advice cues test the same deference on data.

## Statistical knowledge of LLMs (no planted traps)

StatQA ("Are Large Language Models Good Statisticians?", NeurIPS 2024 Datasets & Benchmarks);
QRData (arXiv:2402.17644); StatABench (arXiv:2606.22977); StatFormBench (arXiv:2609.01982);
TimeSeriesExam (arXiv:2410.14752).

## Data-analysis agents: task success and reliability

BLADE (arXiv:2408.09667); DataGovBench, "Data Analysis in the Wild" (arXiv:2607.06482); RADAR / "Fail Loudly"
(silent failures of a data-science agent, 2026); Ambig-DS (task-framing ambiguity, arXiv:2605.09698);
"Many AI analysts, one dataset" (PNAS 2026); ForkSCOPE (arXiv:2609.12438).

## AI scientists and error control

"The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems" (arXiv:2509.08713;
data leakage, metric misuse, post-hoc selection bias); POPPER, agentic sequential falsification with Type-I
error control (arXiv:2502.09858).

## Confidence of tool-using agents

"The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents" (ACL 2026);
Uncertainty quantification for LLM agents, survey (arXiv:2609.07395).

## Statistics background for the traps

Granger & Newbold (1974) spurious regression; Newey & West (1987) HAC standard errors; Hurlbert (1984)
pseudo-replication; Harvey, Liu & Zhu (2016) data snooping in finance; Simpson (1951); Kaufman et al. (2012)
leakage in data mining; Higham (2002) and Qi & Sun (2006) nearest correlation matrix; Cochran,
*Sampling Techniques*, two-stage sampling (allocation analysis).
