# SIREN paper release v0.1.0

This first public release includes:

- Standalone SIREN recognition, eligibility-aware selection and persistent rule registry.
- SIREN-KV and SIREN-Value training/generation, Base and exact PI-v1/v2 controls.
- Matched MiniLM–Discriminant and MaxSim implementations.
- Frozen 82-rule benchmark, separated fit/validation/development/evaluation roles and visible inputs/annotations.
- CPU paper-result reproduction, including routing, eligibility, action completion, retention and expression.
- Optional evidence attachment with cached features, outputs, scores, votes and provenance.

The `siren-paper-evidence-v0.1.0.tar.gz` attachment is 96,381,611 bytes (about 92 MiB). The accompanying SHA-256 file and `artifacts/download.json` bind its contents. Download through `python reproduce/download_artifacts.py`.

No pretrained backbone, trained KV memories, optimizer state, intermediate checkpoints or teacher-probability cache is included. Obtain the backbone separately; use the supplied rule-training recipe when fresh numeric inference is needed. Historical measurements can be recomputed without model weights or paid APIs.

The release validates the frozen results and CPU mechanics; it does not claim a new full-scale GPU rerun. See `docs/REPRODUCING.md` for coverage and protocol boundaries.
