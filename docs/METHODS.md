# Methods and interfaces

## Rule lifecycle

A rule stores its ID, condition and guidance, evidence, recognizer state, eligibility and execution state. Installing makes an entry addressable; refining updates its recognizer from fit-side evidence; use selects one eligible rule or returns abstention. Stored memories persist between requests. Generated answers are not automatically appended to the next request.

## Recognition and selection

The SIREN recognizer averages zero-indexed Llama block 12 over the last 60% of interaction tokens inside the chat template. The condition, target-rule label and reference action are not appended to this query. Rule-local keys use initialization, low-evidence blending and a Ledoit–Wolf shrinkage discriminant from four evidence pairs onward. Fit-only calibration sets acceptance thresholds and score standardization.

The selector first removes uninstalled/ineligible entries. Qualification requires a score above its own threshold. Competition then requires the leading qualified score to exceed the qualified runner-up and the background null by the frozen margin. It can abstain even when some rule is top-ranked. The background-null state is fitted from training backgrounds, not evaluation scenes.

MiniLM–Discriminant uses the same fitting examples and learner with frozen text embeddings. MiniLM–MaxSim compares the query with applicable fit examples; its primary calibration excludes positive self-matches, with a separate self-match sensitivity. A source-indexed fit-bank manifest makes the supervision comparison explicit.

## Numerical execution

**SIREN-Value** writes a 4096-dimensional vector once at the last prompt position into block 29's feed-forward down-projection output. Later decode steps receive no repeated write. Its historical recipe and shorter response protocol are separate from the KV experiment.

**SIREN-KV** compiles the rule-text prefix into native, post-RoPE attention keys and values at every layer. It freezes the initial arrays and their row-wise RMS scales, then fits bounded offsets:

```
K = K0 + 0.5 * sK * tanh(UK)
V = V0 + 0.5 * sV * tanh(UV)
```

The first up to eight non-EOS action tokens receive cross-entropy supervision. Remaining positions, including EOS, receive equally weighted cross entropy and teacher-to-student KL. The teacher is the original frozen model with the same rule prefix, evaluated on the existing reference action; this recipe does not generate additional training text. Rule-local fitting uses Adam, 25 steps and learning rate 0.01. The full implementation specifies example weighting and the 0.01 bounded-offset penalty.

The initialized prefix remains in the attention cache during generation. Cached keys must not receive RoPE twice; suffix positions begin after the prefix. Prefix K/V consume attention positions and storage. No latency or memory-cost advantage over cached text prompting is claimed.

## Text execution controls

PI-v1 supplies the selected rule's condition and guidance as text. PI-v2 adds instructions to write only the next assistant reply without invented dialogue turns. Neither template contains demonstrations. The two controls share the SIREN selection result; they are not independent text-retrieval systems. Base generation receives no rule intervention.

## Execution protocols

1. **Known-rule:** the evaluator explicitly supplies the target rule. This measures execution conditional on the correct rule.
2. **Selected-rule:** the runtime receives only visible interaction text, then uses the rule chosen by the selector, or the base path on abstention.
3. **Sequential retention:** prior execution states and recognizer stay fixed while new execution entries are installed. Evaluate all 666 independent inputs at 83 library sizes.

Recognition refinement and eligibility use a separate 492-snapshot evidence replay. They must not be conflated with the frozen-recognizer execution protocol.

## Validation level

Release assembly validates numerical selection, masking, loss and cache mechanics on CPU and checks frozen result accounting. It does not rerun the full Llama-3.1-8B GPU campaign. The pin set is tested with the release's CPU checks; fresh 8B generation can differ with hardware, numeric precision or dependency changes. Original evidence and model/config identities are included in the result bundle.
