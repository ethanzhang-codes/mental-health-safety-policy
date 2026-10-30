# Final policy-model reassessment

## Outcome

Under a safety-first objective, multinomial logistic regression is the stronger base-policy candidate for the evaluated final pipeline.

This conclusion is based on a matched comparison in which logistic regression and random forest used:

- the same train, validation, and clinician-test partitions;
- the same high-recall `B_pos_weight_x2` Confirm progression features;
- the same Alert, category, and causal progression inputs;
- the same `light_safety` guardrail; and
- the same four-level proxy policy target.

## Matched results

| Split | Model | High-risk unsafe continuation | Over-escalation | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Logistic regression | **0.70%** | 12.25% | 0.4045 | 0.6336 |
| Validation | Random forest | 2.10% | **12.02%** | **0.4087** | **0.6444** |
| Test | Logistic regression | **18.72% (88/470)** | 16.03% (613/3,824) | 0.3769 | 0.6097 |
| Test | Random forest | 22.98% (108/470) | **12.24% (468/3,824)** | **0.3870** | **0.6477** |

On test, logistic regression reduced unsafe continuation by 4.25 percentage points, corresponding to 20 fewer unsafe Level-0 continuations among 470 high-risk turns. It increased over-escalation by 3.79 percentage points, corresponding to 145 additional Level-2/3 actions among 3,824 low-risk turns.

The paired dialogue-level bootstrap estimated:

- unsafe-continuation difference, logistic minus random forest: **−4.25 points**, 95% CI **[−6.22, −2.44]**;
- over-escalation difference: **+3.79 points**, 95% CI **[+2.98, +4.64]**; and
- Macro-F1 difference: **−1.01 points**, 95% CI **[−1.55, −0.46]**.

## Interpretation

Random forest was historically selected under a constraint-and-tiebreaker rule: once a candidate passed the validation unsafe-continuation ceiling, lower over-escalation was prioritized. It was not selected because it had the lowest validation unsafe-continuation rate.

The matched final-pipeline comparison shows that neither model is uniformly superior:

- logistic regression is better at avoiding unsafe continuation;
- random forest produces fewer unnecessary high-level escalations and slightly better Macro-F1 and accuracy.

If high-risk unsafe continuation is the primary criterion and over-escalation is a secondary constraint, logistic regression should be carried forward as the preferred research candidate. If operational burden is prioritized after merely passing a safety ceiling, random forest remains consistent with the historical selection rule.

## Reporting recommendation

Do not silently replace the historical result. Report this as a policy-model reassessment:

> In a matched post-hoc comparison using the final high-recall Confirm features and identical light-safety guardrails, logistic regression reduced held-out high-risk unsafe continuation from 23.0% to 18.7%, while increasing over-escalation from 12.2% to 16.0%. This identifies logistic regression as the stronger safety-first candidate, although the tradeoff requires prospective validation.

## Limitations

- The policy target is a label-derived proxy, not clinical escalation ground truth.
- The same held-out test split had already been examined during the broader exploratory project, so this is a reassessment rather than a pristine confirmatory test.
- The five logistic-regression runs are identical because the fitted solver is deterministic under this configuration. Random-forest variability was negligible for the reported rates.
- The paired bootstrap quantifies sampling uncertainty in this dataset; it does not establish clinical effectiveness.
- Any production or clinical claim requires external data, subgroup analysis, calibration, human review, and prospective evaluation.
