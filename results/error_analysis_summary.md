# Error Analysis Summary

The final policy has 108 high-risk/Confirm-positive Level-0 test failures. The redacted export omits current text, preceding raw context, and analyst free text. IDs and numeric features are retained for private reproducibility; remove IDs too if public-release terms require it.

| category | count | percent |
| --- | --- | --- |
| indirect_or_ambiguous | 78 | 72.22222222222221 |
| important_evidence_earlier | 63 | 58.333333333333336 |
| gradual_progression | 53 | 49.074074074074076 |
| unusual_wording | 50 | 46.2962962962963 |
| other | 40 | 37.03703703703704 |
| calibration_or_confidence | 38 | 35.18518518518518 |
| label_ambiguity | 15 | 13.88888888888889 |

Sixty-three cases (58.33%) were annotated as likely requiring conversation history. Median Confirm probability is approximately 0.0000245, so these are usually not near-threshold misses. The categories and “requires history” judgments come from the existing heuristic/manual analysis script and should not be represented as independent clinician adjudication. Categories overlap, so percentages do not sum to 100%.
