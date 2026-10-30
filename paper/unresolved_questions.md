# Unresolved Questions

## Questions requiring a decision from the author

- Target venue, page limit, citation style, author order/affiliations, anonymization, acknowledgments/funding.
- Whether AnnoMI belongs in the main paper or preliminary appendix.
- Whether to call Level 3 'urgent escalation' while emphasizing it is an offline label.
- Whether code, raw outputs, and checkpoints will be released.

## Missing experimental information

- Exact training wall times and controlled inference latency.
- Multi-seed final CRADLE pipeline and confidence intervals across training seeds.
- A validation-fitted calibration evaluation for final CRADLE Confirm.
- Prospective, external, subgroup, and out-of-distribution evaluation.

## Missing implementation information

- Exact package versions and clean-machine environment.
- Model revision hashes and complete parameter counts.
- Whether all narrative run commands match the actual cluster invocations.

## Missing dataset documentation

- Authoritative CRADLE/AnnoMI release versions, download dates, URLs, licenses, and redistribution terms.
- Confirmation that inferred split-local dialogue IDs preserve source-level non-overlap.
- Formal provenance of synthetic/structured train/validation and clinician test labels.

## Missing citation information

- Complete citations for datasets, encoders, calibration, uncertainty, crisis detection, chatbot safety, and interpretable escalation.

## Missing ethical or administrative information

- IRB/exemption determination, consent/data-use basis, storage/access controls, de-identification review.
- Demographic availability and fairness review.
- Institutional policy on describing emergency/clinical implications.

## Experiments beneficial but not essential

- Context-concatenated Confirm baseline; category-specific temporal models; selective-risk curves; rationale audit.

## Experiments required before a particular claim can be made

- Clinical effectiveness requires prospective expert-designed validation.
- Robustness requires multi-seed and external-domain evaluation.
- Uncertainty-aware final-system claims require selected uncertainty integration and held-out uncertainty-quality tests.
- Deployment/readiness claims require human factors, monitoring, privacy/security, localization, and intervention workflow validation.
