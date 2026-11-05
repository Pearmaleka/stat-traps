# StatTraps

**Do LLM data-analysis agents know when the analysis is wrong?**

LLM agents that write and run code can now analyse a dataset end to end. When the obvious analysis is
*statistically invalid* — two random walks that look related, autocorrelated data treated as independent,
the best of 40 trading signals tested as if it were the only one — does the agent notice? Does it follow the
user into the mistake? And does it *know* when it is wrong?

StatTraps is a small, controlled benchmark built to answer this, with every instance generated from a seed and
an exact ground truth.

> **Status (v0.1, November 2026):** design pre-registered in [`docs/DESIGN.md`](docs/DESIGN.md); generators,
> agent harness and scoring validated offline (below). Model results: in progress.

## Design in one table

Nine families of traps in five groups. Every instance comes in two **worlds** — *trap-active* (the obvious
analysis gives the wrong answer) and *trap-inactive* (it gives the right one) — so an agent cannot score well by
always answering "no effect". Every instance is asked twice: with a **neutral** request and with the user's
**cue** (a suggested method, a false claim about the data, or bad advice), to separate missing statistical
knowledge from deference to the user.

| Group | Family | Trap-active world | The obvious (invalid) analysis | User cue |
|---|---|---|---|---|
| Dependence | `SPUR` | two independent random walks (stock prices) | regress one price level on the other | "use a regression" |
| | `DEP` | autocorrelated daily sales, no real change | before/after t-test | "compare averages with a t-test" |
| | `PSEUDO` | 12 classes × 50 students, class effects | student-level t-test | "use a t-test on the students" |
| Multiplicity | `SNOOP` | 40 candidate trading signals, none predictive | test the best one at 5% | "find the strongest signal and test it" |
| Confounding | `SIMPSON` | treatment given mostly to severe patients | compare overall recovery rates | "compare the overall rates" |
| Leakage | `LEAK` | a column recorded after the outcome | use every column, report CV AUC | "use all the columns" |
| Data integrity | `FREQ` | prices are weekly | annualise with √252 | "these are daily prices" (false) |
| | `SPLIT` | unadjusted 2-for-1 stock split | volatility of raw returns | "prices are split-adjusted" (false) |
| | `GAPS` | staggered listings, 20% missing prices, non-PSD covariance | forward-fill, then estimate risk | "just forward-fill the gaps" |

The answer file always includes the agent's **confidence**, and the final message is checked for whether the
agent **noticed** the issue. For the four inferential families the agent is also evaluated as a **statistical
test** (false-positive rate and power), next to a naive and a valid pre-specified test on the same instances.

## Trap strength (before any LLM is run)

Share of instances where a pre-specified analysis reaches the correct answer
(`python -m stattraps.calibrate --n 100`; 100 fresh instances per world):

| Family | naive: trap-active | naive: trap-inactive | valid: trap-active | valid: trap-inactive |
|---|---|---|---|---|
| `SPUR` | 9% | 100% | 93% | 100% |
| `DEP` | 25% | 100% | 97% | 98% |
| `PSEUDO` | 35% | 100% | 100% | 95% |
| `SNOOP` | 12% | 100% | 97% | 99% |
| `SIMPSON` | 0% | 100% | 100% | 100% |
| `LEAK` | 0% | 100% | 100% | 100% |
| `FREQ` | 0% | 100% | 100% | 100% |
| `SPLIT` | 0% | 100% | 100% | 100% |
| `GAPS` | 11% | 100% | 100% | 100% |

A valid analysis scores close to 100% in both worlds; the naive one fails in the trap-active world.

## How it works

```
stattraps/
  families/         one module per trap: generator (seed, world) -> data + truth + reference decisions,
                    prompts (neutral / cue), scoring, valid & naive reference solutions
  suite.py          builds instances/<FAMILY>_w<world>_<i>/ (data files the agent sees + truth it does not)
  agent.py          minimal agent: one neutral system prompt, one tool (run_python, persistent session),
                    any OpenAI-compatible API, provider-specific fields passed back untouched
  run.py            multi-model runner: resumable, per-model budget cap, rate limits, balanced ordering
  analyze.py        balanced accuracy, agent-as-a-test, cue effects, awareness, calibration, figures
  allocation.py     variance components: how to split a budget between new instances and repeated runs
  calibrate.py      trap strength of naive vs valid analyses
  validate.py       offline end-to-end validation with a scripted fake model (no API key)
docs/DESIGN.md      pre-registered design, hypotheses and analysis plan
docs/RELATED_WORK.md
```

The harness is deliberately minimal (~150 lines) so that results measure the model rather than an agent
framework. It never mentions pitfalls or checking the data; the optional `--skill` condition appends a
statistical checklist (`stattraps/skill.md`) to the system prompt.

## Validation

- `pytest -q tests` — deterministic generation, no trap-revealing words in neutral prompts, scoring of true
  and wrong answers (35 tests).
- `python -m stattraps.validate` — a scripted fake model sends the valid and the naive reference solution of
  every instance through the **real** agent loop (kernel, tool calls, `answer.json`, scoring): 180 runs, and the
  score assigned by the pipeline equals the generator's reference decision in 180/180.

## Quick start

```bash
pip install -r requirements.txt
python -m stattraps.suite --out instances          # 90 instances (5 per world per family)
python -m stattraps.validate                       # offline check, no API key
cp configs/models.example.json configs/models.json # edit model ids
cp .env.example .env                               # add your API keys (never commit .env)
python -m stattraps.models --list deepseek-v4-pro  # see available model ids
python -m stattraps.run --use deepseek-v4-pro --k 2 --budget-usd 3
python -m stattraps.analyze                        # report/summary.md + report/figures/
python -m stattraps.allocation                     # RQ5: instances vs repeated runs
```

On Windows: `scripts\setup_and_check.bat`, then `scripts\run_models.bat deepseek-v4-pro 2 3`.

Any OpenAI-compatible endpoint works (DeepSeek, Gemini via its OpenAI-compatible endpoint, OpenRouter, local
servers such as vLLM or Ollama). Default run per model: 9 families × 2 worlds × 5 instances × 2 prompts ×
2 repetitions = 360 runs.

## Limitations

- Synthetic data and one-shot requests; real analyses are messier and interactive.
- Prompts are in English; one minimal harness (results under other agent frameworks may differ).
- "Noticed" is detected with keyword patterns and should be spot-checked by hand.
- The checklist (`--skill`) was written by people who know the traps: its effect is an upper bound.

## Related work

See [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). Closest: agents as statistical tests in planted-signal
worlds (Liu 2026), CausalPitfalls (2025), sycophancy in financial agents (2026).

## License

MIT — see [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
