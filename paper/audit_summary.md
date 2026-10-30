# Final Audit Summary

## Fully verified

- AnnoMI was preliminary/prototype-only in the audited safety pipeline; its reported classification metrics use pseudo-labels.
- CRADLE is the main Alert/Confirm/category/progression and policy evaluation source.
- Separate binary Alert and Confirm models exist; final Confirm is current-turn-only.
- `B_pos_weight_x2` doubles the training-derived positive weight and was selected on validation only.
- Four policy levels, including offline urgent escalation, are implemented.
- Final candidate: safety-constrained random-forest policy + `light_safety` guardrail + B Confirm.
- Main test rates and exact counts: unsafe continuation 140/470 (29.79%) → 108/470 (22.98%); over-escalation 455/3,824 (11.90%) → 468/3,824 (12.24%).
- v3 was not selected because no validation candidate improved unsafe continuation under the 0.20 over-escalation constraint.
- Remaining misses have very weak upstream probabilities in the saved error analysis.
- No custom cross-attention module exists; temporal embeddings/features are used.

## Partially verified

- Conversation-level split integrity is supported by separate source splits and inferred IDs, but source-level identity overlap cannot be independently ruled out from split-local inferred IDs.
- Five-seed ensemble and temperature scaling are verified for the older severity prototype, not final CRADLE policy.
- Logs support several GPU/model runs, but complete hardware/software/runtime provenance is absent.
- “Final” means selected research candidate, not deployment-ready system.

## Unsupported

Clinical validation, diagnosis, real-world effectiveness, deployment readiness, real emergency initiation, final-system ensemble uncertainty, statistical robustness across CRADLE seeds, and fairness/generalization claims.

## Primary result and interpretation

The primary result should be the isolated old-to-B Confirm swap in the safety-constrained policy on CRADLE clinician test, reported with both unsafe continuation and over-escalation. It is a single-seed held-out dataset result. The paired audit is post hoc; avoid unqualified significance language.

## Paper readiness

The project is ready for a carefully bounded draft focused on offline methodology and empirical safety tradeoffs. Critical issues before submission are authoritative dataset/citation/license details, IRB/administrative statements, environment pinning, and a decision about how prominently to disclose exploratory test reuse. Multi-seed CRADLE evaluation is strongly recommended before a robust empirical claim.

## Files for a paper-writing assistant

Provide `paper_facts.md`, `paper_facts.json`, `audit_summary.md`, `source_inventory.md`, `unresolved_questions.md`, all `methods/`, `datasets/`, `models/`, `results/`, `tables/`, `figures/`, `ethics/`, `references/`, and `reproducibility/` files. Do not provide raw CRADLE/AnnoMI text or model checkpoints by default.

## Package counts

- Repository files inspected/inventoried: 428
- Verified experimental conditions: 11
- Generated publication tables: 12
- Generated figures: 12
- Unresolved questions: 23
