# Release verification (2026-10-08)

- Local CPU suite: **49 passed**, including tiny randomly initialized Llama training, cached generation and true routing-to-delivery integration.
- Fresh GitHub checkout and isolated Python 3.13 environment: installation succeeds; **38 passed, 2 skipped** without optional model libraries. The offline example and complete primary result reconstruction succeed.
- Four selectors: **9,560/9,560** selected rules and abstention reasons match the archive. All three MiniLM variants were refitted from the released evidence/features; the archived SIREN fitted state was replayed.
- Real tokenizer comparison: all **82 prefixes, 491 training records and 164 validation records** match saved token IDs, boundaries and EOS supervision.
- **90 manifest entries** verify; **2,676 behavior vote packets** reconstruct their saved labels.
- Primary routing, recognition, eligibility, execution, retention and preference statistics reproduce from released records.
- Export preserves input/reference strings and known data issues; separate input files contain only `id` and `stream`.
- Publication scan found no credentials or unintended private paths. No backbone weights, optimizer states or teacher-probability caches are distributed.

A pre-publication review caught an aggregation mistake: the source field `n_offdiag` denotes possible off-diagonal rule pairs, not wrong decisions. The released result code instead reconstructs wrong decisions from each rule's steal rate and applicable-context count, verifies the paper's 18→3 peak and 1,313→485 burden, and includes a regression test. Historical evidence itself was unchanged.

This verifies release assembly and CPU algorithm mechanics, not a new full 8B GPU campaign. No paid API calls were made. See `validation.json` and `REPRODUCING.md` for scope.
