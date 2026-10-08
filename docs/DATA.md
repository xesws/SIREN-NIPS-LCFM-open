# Frozen benchmark and evidence

The release targets the LCFM camera-ready paper, version 1.6. It exports the
existing inputs, decisions, responses, and judgments. It does not generate new
data, repair historical examples, or rerun experiments.

## Data files

`data/rules.json` contains 82 rule objects with `rule_id`, `condition`, and
`guidance`. Ten are seed rules and 72 are expanded rules. The excluded candidate
`r033` is not in this roster.

| File/group | Contents |
|---|---|
| `recognition/inputs.jsonl` | 5,462 records with **only** `id` and `stream` |
| `recognition/annotations.jsonl` | Matching IDs, fit/eval role, view, target rule and applicable/hard-negative/background kind |
| `recognition/fit_banks.json` | Exact positive and negative ID banks for each final recognizer |
| `execution/train.jsonl` | 491 SIREN-KV fit records: ID, rule ID, stream, reference action and evidence index |
| `execution/validation.jsonl` | 164 validation records in the same format |
| `execution/inputs.jsonl` | 666 inference records with **only** `id` and `stream` |
| `execution/annotations.jsonl` | Separate rule IDs, reference behaviors, conditions and development/main/seed membership |
| `execution/value_train.jsonl` | Original SIREN-Value seed recipe: 10 rules × 8 examples, with its historical thought and action |
| `execution/splits.json` | Explicit sample-ID and rule-ID lists |
| `protocols/` | 492 recognition measurement snapshots and 83 execution installation sizes |

The recognition fit pool contains 82 × (20 applicable + 16 hard-negative)
examples and 120 calibration backgrounds: 3,072 records. The final classifier
uses 16 own applicable examples, 16 own hard negatives, 120 backgrounds and
20 applicable examples from each of the other 81 rules. Thus the negative bank
has 1,756 records. All bank IDs have a fit role; no evaluation ID enters fitting.

| Recognition evaluation panel | Applicable | Hard negatives | Background |
|---|---:|---:|---:|
| Seed | 90 | 90 | — |
| Original expanded | 432 | 216 | — |
| Turn-removed paired view (`transformed`) | 432 | 216 | — |
| New contexts (`fresh`) | 576 | 288 | — |
| Background | — | — | 50 |

These are 2,390 evaluation records across panels, not 2,390 independent semantic
scenarios: original and turn-removed inputs are paired views. Hard negatives
are negative relative to their target rule. Another rule's applicable example
is a contrastive fit negative, not necessarily a system-level background.

Execution is divided into development (14 rules, 112 scenes), main (58 rules,
464 scenes) and seed (10 rules, 90 scenes). All 666 concern applicable situations
of learned rules. They do not constitute a locality test. The separate routing
panels contain non-applicable situations. An evaluation rule still has its own
fit examples; it was withheld from architecture/template development, not from
rule-specific fitting. For SIREN-KV, the 14 development rules have 84 fit and
28 validation examples, and the other 68 have 407 and 136.

## Runtime and annotation boundaries

The normal inference API consumes `inputs.jsonl` only. IDs are join keys and
must never be embedded, formatted into the prompt, or treated as features.
Rule IDs and reference actions in annotations are evaluator information.
Training may use reference actions from the fit file. Correct-rule execution
is an explicitly separate experimental condition that provides the target
rule; ordinary selective routing does not receive it.

The data are controlled synthetic fixtures, and the benchmark informed repeated
development. Original inputs sometimes contain action cues. The turn-removed
view removes a trailing assistant turn and is not a claim of complete
decontamination. Its known cross-label duplicate, `transformed:r051-p03` and
`transformed:r051-h02`, is preserved and recorded in `recognition/audit.json`.
Whitespace, input wording, reference actions and frozen labels are preserved.

## Sequential protocol

This paper uses a **known rule collection**: rule descriptions and cross-rule
contrastive examples are available before replay. Rule installation and own
evidence arrive progressively. Recognition replay has 492 measurement
snapshots. The execution experiment fixes its recognizer and progressively
installs 82 execution memories, measuring 666 independent contexts at sizes
0 through 82. A generated response is not appended to another test input.
Future execution memories are unavailable before installation. This is not
learning truly unknown future rules, and the data/export names preserve that
distinction.

## Downloadable frozen evidence

`artifacts/frozen/` is excluded from source Git. The separately downloaded
bundle supports CPU-only reconstruction of the paper:

- `routing/`: per-case predictions, raw/standardized score matrices and states.
  Score arrays use `raw`, `z`, `thresholds`, `null_raw`, and `null_z`; the index
  file fixes the 2,390-example and 82-rule order.
- `recognition/`: complete per-snapshot/per-rule measurements with eligibility
  off/on and an alternative replay order.
- `execution/outputs.jsonl`: deduplicated original response text, tokens,
  selected rule, behavior labels, and judge-packet references. `cases.jsonl`
  maps each method/condition/sample to an output; `stages.jsonl` maps each
  installation size to outputs and routing decisions.
- `execution/judge_packets.jsonl`: the individual frozen votes, cited evidence,
  rationales and final labels. `positive` means a rule-consistent response,
  `action` means actual core action completion, and `repetition` means collapse.
  These historical field names have explicit public definitions.
- `expression/`: all paired responses, merged judgments, per-vote request
  payloads, evidence, and published analyses for PI-v1 and PI-v2. Nested
  `numeric`/`pi` labels are historical provenance; each row names its public
  comparison. No invalid quote is silently promoted to verified evidence.
- `features/`: optional cached layer-12 and MiniLM features in NPZ files with
  Unicode `ids` and a numeric `features` matrix. They are an evaluation/fitting
  cache, not the normal input API. All rows retain separate role annotations.
- `supplement/`: source records supporting the historical Value and matched
  recognition panels; names are descriptive, original fields are preserved.

The SIREN null direction was built from one archived background extraction and
calibrated on another. Both are preserved in `routing/siren_keys.npz`; the
reusable `siren_null.json` records `key`, `mean` and `std`. The execution selector
is separately recorded because it uses fresh float32 features. Do not conflate
that record with cached recognition measurements. Final and initialization
recognition keys are provided for checking the frozen implementation.

No backbone weights, KV weights, optimizer state, teacher probability cache,
secrets, logs or machine configuration are in this evidence export. The
historical paid API verdicts are frozen evidence; rerunning a changed API is
not guaranteed to give identical verdicts.

## Provenance and author-side export

`artifacts/source_map.json` records relative archive identities and SHA-256
hashes of source and exported files. Source identities are provenance, not
runtime dependencies. `data/export_audit.json` gives expected counts and missing
optional sources. The frozen bundle's `manifest.json` binds every exported
file except itself. Consumers never need the private source archive.

An author with the archived inputs can rebuild the bundle using NumPy:

```bash
python tools/export_paper_data.py \
  --source-repo /path/to/research-source \
  --results-root /path/to/canonical-results \
  --output-root /path/to/public-checkout
```

The command only reads the two source roots. It writes the public data and
staging asset directory, invokes no model/API, and makes no Git changes itself.
The independent data tests check input allowlists, disjoint roles, historical
known issues, schedules, hashes and evidence references.
