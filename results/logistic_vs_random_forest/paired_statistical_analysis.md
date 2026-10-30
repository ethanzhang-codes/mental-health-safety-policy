# Paired statistical analysis

Uncertainty was estimated with 10,000 paired cluster-bootstrap replicates over dialogues. Differences below are logistic regression minus random forest; negative unsafe-continuation differences favor logistic regression, while positive over-escalation differences indicate additional burden.

## Test split

- Unsafe continuation: logistic regression 0.1872 (88/470) versus random forest 0.2298 (108/470). Difference -0.0425, 95% bootstrap CI [-0.0622, -0.0244].
- Over-escalation: logistic regression 0.1603 (613/3824) versus random forest 0.1224 (468/3824). Difference 0.0379, 95% bootstrap CI [0.0298, 0.0464].
- Macro-F1 difference: -0.0101, 95% bootstrap CI [-0.0155, -0.0046].

## Validation split

- Unsafe continuation: logistic regression 0.0070 versus random forest 0.0210.
- Over-escalation: logistic regression 0.1225 versus random forest 0.1202.

## Decision interpretation

Logistic regression provides the lower unsafe-continuation rate on both validation and test under the matched final guardrail. Random forest provides lower over-escalation and slightly higher Macro-F1. Choosing between them therefore requires an explicit safety-versus-burden objective; the result does not support claiming that random forest is uniformly superior.
