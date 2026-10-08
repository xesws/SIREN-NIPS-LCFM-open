# SIREN

**Evidence-Coupled Rule Editing for Selective Memory Use**
Tangyi Qian · Yuan Chang
NeurIPS 2026 Workshops: LCFM and CL4FMAgents

[LCFM paper](https://openreview.net/forum?id=RmMFU7QAnn) · [CL4FMAgents paper](https://openreview.net/forum?id=NEyGbpB93a) · [Data](docs/DATA.md) · [Reproduction](docs/REPRODUCING.md) · [Methods](docs/METHODS.md)

SIREN stores behavioral rules as individually maintained memories. Each entry combines a condition, a requested action, a learned applicability recognizer, an activation eligibility state, and an execution state. **Install**, **Refine**, and **Use** maintain rules across independent requests. A request selects one eligible rule or abstains and uses the base model.

This release contains the paper's selective-use mechanism, SIREN-Value and SIREN-KV execution, matched MiniLM selectors, PI-v1/v2 controls, frozen benchmark data, and a separately downloaded result-evidence bundle. Public modules are organized by mechanism rather than research-round identifiers.

## Quick start

Python 3.11–3.13 is required. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[report,dev]'
python examples/select_rule.py
python -m pytest
```

The numerical selection example and default tests do not download language models or call paid APIs. Optional model tests run when `.[model]` is installed and use tiny randomly initialized local models.

## Reproduce the paper's measurements

```bash
python reproduce/download_artifacts.py
python reproduce/paper_results.py --artifacts artifacts/frozen --output outputs/paper
```

The evidence bundle contains saved decisions, generated text and frozen judgments. The second command recomputes measurements from these records on CPU; it does not regenerate answers or buy new judgments. Expected counts are checked against the paper. See [reproduction scope and commands](docs/REPRODUCING.md).

## Run or train the model

Install `.[model]` and obtain an authorized copy of `meta-llama/Llama-3.1-8B-Instruct` separately. Model paths and devices are explicit arguments; the code has no dependency on a particular cloud provider or private results directory. See the model examples and [method guide](docs/METHODS.md). Full 8B training and generation are GPU workloads and are **not** started by the quick start or result-reproduction command.

The original backbone weights, intermediate training checkpoints and teacher-probability caches are not distributed. Final trained memories are not required to recompute the paper's frozen results; this initial release provides the recipe to train them locally rather than an automatic multi-gigabyte model-state download.

## What the evaluation measures

- **Condition recognition:** per-rule ROC AUC as evidence accumulates.
- **Complete selection:** correct rule, wrong rule, abstention, correct rejection and false activation. Balanced exact routing averages correct-rule recall and correct rejection.
- **Execution:** action completion, rule-consistent response and repetitive degeneration, with the correct rule supplied, after a correct selection, and end to end.
- **Retention:** behavior across 83 installed-library sizes, using independent requests and a fixed recognizer.

The experiments replay a **known rule collection**. Cross-rule training evidence is available before recognition replay. Execution retention fixes the recognizer and installs rule-local execution memories sequentially. The 666 execution scenes are applicable scenes; locality is measured using separate hard-negative/background panels. These settings are not a benchmark of genuinely unknown future-rule arrivals.

The benchmark includes generated data and known input-quality limitations. This release preserves the historical inputs and judgments; it does not silently clean the benchmark or relabel outputs. Newly generated model answers and API judgments may differ from the frozen paper records.

## Repository layout

`src/siren/` contains memory, recognition, routing, execution, baselines and evaluation modules. `data/` separates visible inputs from annotations and split/protocol manifests. `configs/` and `prompts/` preserve experiment settings. `reproduce/` reconstructs results, `examples/` provides small uses, and `artifacts/` records downloadable evidence and checksums. [Release checklist](docs/RELEASE_TODO.md).

## Attribution

Please cite the paper using [CITATION.cff](CITATION.cff). Third-party models and method sources are identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are obtained under their upstream terms.

Project-authored code is released under [MIT](LICENSE). Benchmark attribution
and data terms are described in [data/LICENSE.md](data/LICENSE.md).
