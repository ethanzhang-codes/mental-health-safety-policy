# Reproducibility Checklist

- [x] Existing train/validation/test CSVs and prediction outputs found.
- [x] Seeds and selected thresholds recorded for core CRADLE components.
- [x] Conversation-order leakage audits found.
- [x] Final policy outputs and validation-selection reports found.
- [x] 428 relevant source/artifact files hashed in `source_inventory.csv`.
- [ ] Authoritative dataset download/version/license records found.
- [ ] Complete pinned environment found (none found).
- [ ] End-to-end clean-machine reproduction demonstrated.
- [ ] Multi-seed CRADLE final pipeline demonstrated.
- [ ] Checkpoint/model-card licensing and public-release review completed.

The existing results are reconstructable from saved outputs, but full retraining reproducibility is incomplete because dependency pinning, dataset acquisition/versioning, hardware details, and some run orchestration are undocumented. Model checkpoints are large and should use a model registry or Git LFS rather than ordinary Git. Raw dialogue and derived rows containing text must not be published until dataset terms and privacy requirements are confirmed.
