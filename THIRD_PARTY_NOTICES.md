# Third-party methods and assets

This public package extracts SIREN's project-authored implementations and provides standalone numerical operations. It does not redistribute the private research repository or pretrained backbone weights.

| Asset / method | Use and source |
|---|---|
| Llama-3.1-8B-Instruct | Frozen backbone. Obtain separately from `meta-llama/Llama-3.1-8B-Instruct` under the Llama 3.1 Community License. Paper model revision: `0e9e39f249a16976918f6564b8830bc894c89659`. |
| all-MiniLM-L6-v2 | Text representation baseline, distributed by sentence-transformers under Apache-2.0. Model revision is recorded in the configuration/evidence. |
| bge-m3 | Historical rule-text retrieval comparison; upstream model has its own MIT terms. No weights included. |
| HoReN | Source of the codebook editing/scoring approach (arXiv:2605.08143), acknowledged by the paper. |
| EasyEdit | The historical research backend included an EasyEdit-style HoReN trim via a third-party prototype. Its provenance is recorded, but that vendored tree lacks a separate license notice. It is not copied into this public package. Standalone selector/hook/token-packing code implements the SIREN equations and preserves the tested prompt/EOS behavior. |
| DeepSeek / Claude | Historical generated benchmark materials and model-judge outputs are identified as such. The bundle includes prompts, outputs and labels, not provider credentials or model weights. New API use is optional and subject to provider terms. |

Python dependencies retain their upstream licenses. Project licensing does not override any third-party model or data rights. Benchmark text and frozen outputs are research artifacts and must be interpreted with the provenance and limitations in `docs/DATA.md`.

Historical experiment IDs and source-file hashes appear only in the release provenance map, allowing comparisons with the paper without exposing private machine paths.
