# Metric Definitions

| metric | formula | denominator_population | level | direction | implementation |
| --- | --- | --- | --- | --- | --- |
| Accuracy | correct predictions / all examples | all labeled turns | turn | higher | sklearn accuracy |
| Macro-F1 | unweighted mean class F1 | all policy classes | turn | higher | sklearn macro F1 |
| Weighted-F1 | support-weighted mean class F1 | all policy classes | turn | higher | sklearn weighted F1 |
| AUROC | area under ROC ranking curve | masked task turns | turn | higher | sklearn roc_auc_score |
| AUPRC / average precision | precision-recall ranking summary | masked task turns | turn | higher | sklearn average_precision_score |
| High-risk unsafe continuation | count(high-risk AND policy level 0) / count(high-risk) | Confirm or any category-positive user turns | turn | lower | policy evaluate_policy |
| Alert unsafe continuation | count(Alert AND level 0) / count(Alert) | Alert-positive user turns | turn | lower | policy evaluate_policy |
| Confirm unsafe continuation | count(Confirm AND level 0) / count(Confirm) | Confirm-positive user turns | turn | lower | policy evaluate_policy |
| Over-escalation burden | count(no-risk AND level >=2) / count(no-risk) | no Alert, Confirm, or category target | turn | lower | policy evaluate_policy |
| Urgent escalation precision | count(level 3 AND policy target 3) / count(level 3) | Level-3 policy outputs | turn | higher | policy evaluate_policy |
| Human defer precision | count(level >=2 AND policy target >=2) / count(level >=2) | Level-2/3 outputs | turn | higher | policy evaluate_policy |
| Time to escalation | first user-turn index with level >=2 | each dialogue | conversation | context dependent | policy time_to_level |
| ECE | sum bin fraction * \|bin accuracy - bin confidence\| | predictions with labels | turn | lower | temperature_scaling.py |
| Brier score | mean squared probability error | predictions with labels | turn | lower | reconstructed in audit |
| NLL | mean negative log probability of true class | predictions with labels | turn | lower | temperature_scaling.py |

Under-escalation and correct-escalation are represented through the policy confusion matrix but are not separately named rates in the final reports. No cost-weighted policy metric is reported. Turns-to-Alert/Confirm exist as dataset progression metadata and are evaluation-only; they are not standard final performance metrics. Older AnnoMI `evaluate_framework.py` defines unsafe continuation as high-risk at Level 0 or 1, which differs from the final CRADLE definition (Level 0 only). Do not mix them.
