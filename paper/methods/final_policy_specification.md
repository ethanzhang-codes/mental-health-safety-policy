# Final Policy Specification

## Final candidate

The repository's frozen final candidate is **random-forest learned policy + `light_safety` guardrail + high-recall Confirm `B_pos_weight_x2`**. It is a hybrid, safety-constrained offline policy over four proxy escalation levels. It does not contact humans or emergency services.

| level | action | learned_component | guardrail_summary |
| --- | --- | --- | --- |
| 0 | continue conversation | learned prediction unless guardrail raises it | none |
| 1 | offer supportive resources | learned prediction or light guardrail | minimum 1 for moderate Alert/Confirm/volatility patterns |
| 2 | defer to human support | learned prediction or guardrail | minimum 2 for high Confirm or accumulated Alert patterns |
| 3 | urgent escalation | learned prediction or guardrail | minimum 3 for high Confirm plus suicide/self-harm or escalation evidence |

The supervised `policy_target` is derived from CRADLE Alert, Confirm, and category labels; code explicitly calls it a proxy, not clinical ground truth. Candidate classifiers include logistic regression, random forest, and optional XGBoost. Validation constraints and tie-breakers select the model/guardrail. The final computation is exactly:

`policy_level = max(learned_policy_level_before_guardrail, guardrail_minimum_level)`.

Policy probabilities/features and rule triggers produce deterministic template rationales. No human faithfulness/correctness study is recorded.

## Versions

The initial safety-constrained script is v1; v2 expands/compares guardrail variants and validation constraints. The high-recall run reuses v2 logic while replacing only Confirm predictions. “v3” is a later search over additional hand-written rules. No v3 candidate improved validation high-risk unsafe continuation under validation over-escalation <= 0.20, so no rules were selected and test outputs remained unchanged.

The code records validation-only guardrail selection and no test use for v3 selection. The broader project is exploratory and contains repeated test reports, so “frozen before any test inspection” cannot be established for the entire research history.

## Main test rates

| condition | dataset | split | metric | numerator | denominator | rate | ci95_low | ci95_high | uncertainty_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Safety-constrained + original Confirm | CRADLE-Dialogue | clinician test | high-risk unsafe continuation | 140 | 470 | 0.2978723404255319 | 0.2570263925079332 | 0.3389839543204076 | single seed; dialogue-cluster bootstrap CI |
| Safety-constrained + original Confirm | CRADLE-Dialogue | clinician test | over-escalation burden | 455 | 3824 | 0.11898535564853556 | 0.11177074133160032 | 0.12627218504104498 | single seed; dialogue-cluster bootstrap CI |
| Safety-constrained + B_pos_weight_x2 | CRADLE-Dialogue | clinician test | high-risk unsafe continuation | 108 | 470 | 0.2297872340425532 | 0.19238126138195216 | 0.2670689166924107 | single seed; dialogue-cluster bootstrap CI |
| Safety-constrained + B_pos_weight_x2 | CRADLE-Dialogue | clinician test | over-escalation burden | 468 | 3824 | 0.12238493723849372 | 0.11526112722529 | 0.12977898684825706 | single seed; dialogue-cluster bootstrap CI |
