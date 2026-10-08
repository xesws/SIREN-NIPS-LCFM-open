# Reproduction guide

The initial release is tied to the paper's v1.6 evidence. It distinguishes
recomputing **frozen measurements**, rerunning **inference**, and **training**.
No command in the first section starts a model or calls a judge API.

## 1. Recompute the primary paper results on CPU

```bash
python -m pip install -e '.[report,dev]'
python reproduce/download_artifacts.py
python reproduce/paper_results.py --output outputs/paper
```

Download size and SHA-256 are recorded in `artifacts/download.json`. The frozen
bundle is about 92 MiB compressed and 270 MiB extracted. It contains cached
features, fitted selector states, outputs and judgments; no backbone weights,
optimizer state or teacher-probability tensors. The downloader verifies the
archive and rejects unsafe paths. A matching local download is reused.

The result command verifies the source-data/evidence manifest, recomputes the
measurements below and asserts their frozen counts. It writes `metrics.json`,
`results.md` and three PNG plots. Use `--no-plots` if matplotlib is not installed.
The plots present the original measurements; their layout need not match the
paper PDF pixel for pixel.

| Paper result | Public evidence | Recomputed quantity |
|---|---|---|
| Recognition refinement (Figure 3) | Per-rule independent measurements at each own update | Mean AUC and interquartile range at 0/1/2/4/8/16 pairs |
| Complete selection (Table 1) | Four methods' decisions and target scores on 2,390 rows | Five outcome counts, balanced score, own-rule AUC, paired rule bootstrap |
| Eligibility (Figure 4) | Paired 492-snapshot histories | Wrong-row peak 18→3, burden 1,313→485, delay in correct selection |
| Execution (Table 2) | Deduplicated outputs, case mappings, frozen labels | Supplied-rule 430/430/409; correct-selection 127/128/118; end-to-end 221/221/212 |
| Sequential retention | Output references at 83 library sizes | Per-rule post-installation baseline, final change and worst later decline |
| Expression | Original paired replies and blinded preference judgments | Rule-balanced preference and rule-cluster intervals for both PI templates |

The three execution rows are SIREN-KV, PI-v1 and PI-v2. Main denominators are
464, 142 and 464 respectively. Base action completion is 136/464. The 142
correctly selected cases are a subset of the 464 applicable execution scenes,
not the new-context routing panel used in Table 1.

`artifacts/frozen/supplement/` preserves the older Value negative-result
responses/preferences, component controls, input-view comparisons and
distillation evidence. These are source records and historical analyses;
the main result command does not rerun every appendix training campaign.
`artifacts/source_map.json` identifies every exported input by its historical
relative identity and hash. No private checkout is needed to consume the release.

## 2. Refit matched text selectors using the released feature bank

```bash
python -m siren.recognition fit \
  --data-root data --features artifacts/frozen/features/minilm.npz \
  --method minilm_discriminant --output outputs/minilm_registry.json
python -m siren.recognition evaluate \
  --data-root data --features artifacts/frozen/features/minilm.npz \
  --registry outputs/minilm_registry.json --output outputs/minilm_evaluation.json
```

Repeat with `--method minilm_maxsim` or `minilm_maxsim_self` to test the two
MaxSim calibrations. Fitting is restricted to fit-bank IDs; target annotations
are joined only after label-free evaluation decisions. All three public
refits were checked against the original 2,390 predictions and abstention
reasons per method, with zero mismatches.

For SIREN, `siren.recognition.artifacts.load_frozen_registry(execution=False)`
loads the archived recognition selector. Its predictions on the released
Llama feature bank were also checked with zero mismatches. The optional
`recognition extract` command computes features with a locally supplied model;
fresh features may differ numerically from the archived bank. See `--help`.

## 3. Train a rule and generate a response

Install `.[model]` and obtain the Llama model from its authorized upstream
source. Replace `/path/to/model` below with a local snapshot. The release does
not download it or start a remote GPU machine.

```bash
python examples/train_rule.py \
  --model-path /path/to/model --rule-id r001 \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --output outputs/memories/r001.pt

python examples/generate.py \
  --model-path /path/to/model --method siren-kv \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --checkpoint outputs/memories/r001.pt \
  --input data/execution/inputs.jsonl --sample-id r001-p01
```

This generation example supplies a particular trained rule. It is **not** an
end-to-end routing measurement. The training file contains only the per-rule
fit examples; validation and evaluation scenes are not used to update memory.
Only a final checkpoint is saved. KV uses FP32, a 25-step schedule and a
1,024-token generation cap. It can require roughly the original A40-class
GPU memory budget; reduce the example workload, not precision, when checking
the exact paper recipe.

`--method siren-value` uses the separate original Value recipe, defaults to
BF16 and 96 generation tokens. Its supplied `value_train.jsonl` covers the
10 seed rules' 80 historical fit records. The KV and Value datasets and
protocols differ; running them does not create a controlled interface ablation.

## 4. Actual selection followed by execution

```bash
python examples/agent.py \
  --model-path /path/to/model --delivery pi-v2 \
  --input data/execution/inputs.jsonl --sample-id r001-p01
```

This entrypoint first computes the original request's Layer 12 representation,
then selects or abstains. PI receives the **selected** rule. No target rule is
read from annotations. For numeric delivery, use `--delivery siren-kv` and
`--checkpoint-dir outputs/memories`, with one final file per installed rule.
A missing selected-rule checkpoint is an error, not permission to silently
switch the route or give the generator the answer. `--registry` can provide a
custom installed-rule registry. The default is the frozen execution selector.

## 5. Judgments and reproducibility boundaries

The result bundle retains all behavior votes and their diagnostic evidence.
`siren.evaluation.judging.merge_votes` derives each vote's action/positive/
repetition labels before majority voting, matching the paper. A model's visible
correct plan can count as a rule-consistent response without completed action.
Mechanical repetition overrides both positive and action labels.

To judge a **new** single response, install `.[judge]`, set your own
`DEEPSEEK_API_KEY`, and explicitly run `examples/judge_response.py` with
`--allow-paid-api`. The item must contain `rule_condition`, `required_behavior`,
`dialogue` and `candidate_reply`. Use a new output file. This optional call is
billable, never runs in tests, and does not rewrite the frozen evidence. The
saved prompt/config record the historical model family; continued provider
availability and identical new judgments are not guaranteed.

The release's CPU checks include tiny random Llama training/generation and
cache/causality tests. Historical tokenizer positions were verified against
all 82 prefixes, 491 fit records and 164 validation records. The complete
8B GPU campaign was **not rerun during packaging**.
