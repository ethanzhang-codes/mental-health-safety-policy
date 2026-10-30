# Baselines and Experimental Conditions

| condition | difference | dataset | splits | seed_count | status |
| --- | --- | --- | --- | --- | --- |
| Single-turn Alert | current text only | CRADLE | validation/test | 1 | baseline |
| Temporal Alert hw1/hw4/hw8 | frozen embedding history windows | CRADLE | validation/test | 1 | hw8 selected |
| Original Confirm | standard weighted BCE; threshold .24 | CRADLE | validation/test | 1 | replaced |
| High-recall Confirm A/B/C/D | loss/positive weighting variants | CRADLE | validation/test | 1 | B selected on validation |
| Rule-based policy | hand rules | CRADLE | test | 1 | comparison only |
| Unconstrained learned policy | supervised policy classifier | CRADLE | validation/test | 1 | comparison only |
| Safety-constrained old Confirm | learned + validation-selected guardrail | CRADLE | validation/test | 1 | replaced |
| Safety-constrained B_pos_weight_x2 | only Confirm predictions replaced | CRADLE | validation/test | 1 | final candidate |
| Guardrail v3 search | additional candidate rules | CRADLE | validation then test | 1 | rejected |
| Probability uncertainty ablation | adds entropy/confidence/margin features | CRADLE | validation then test | 1 | rejected |
| Five-seed calibrated ensemble | ensemble + temperature scaling | constructed severity data | validation/test | 5 | prototype only |

Test metrics are descriptive held-out reports, but multiple exploratory model families were evaluated on test. The paper must not describe the best observed test result as a single preregistered selection. B Confirm and v3 decisions explicitly use validation only.
