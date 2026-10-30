# Statistical Analysis

| metric | denominator | old_count | new_count | old_rate | new_rate | absolute_change_percentage_points | relative_change_percent | improved_cases | worsened_cases | paired_exact_mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high-risk unsafe continuation | 470 | 140 | 108 | 0.2978723404255319 | 0.2297872340425532 | -6.8085106382978715 | -22.857142857142854 | 32 | 0 | 4.656612873077393e-10 |
| over-escalation burden | 3824 | 455 | 468 | 0.11898535564853556 | 0.12238493723849372 | 0.33995815899581644 | 2.857142857142862 | 8 | 21 | 0.0241195447742939 |

The main rates are turn-level conditional proportions. Confidence intervals in `policy_results.csv` use 4,000 dialogue-cluster bootstrap resamples, preserving within-dialogue dependence. The paired exact McNemar/binomial calculation compares old/new binary outcomes on the same eligible turns. These are post hoc audit analyses, not preregistered tests. No multiple-comparison correction was applied, and no claim of statistical significance should be made without a prespecified analysis.

The unsafe denominator is all user turns satisfying the implementation's high-risk definition: `target_confirm == 1` or any Confirm-linked target category positive. In these files this is 470 test turns; 140 were Level 0 under the old policy and 108 under the new policy. The no-risk denominator is 3,824 user turns with no Alert, Confirm, or category target; 455 were Level 2/3 under the old policy and 468 under the new policy.

Only one CRADLE seed exists for the selected Alert/Confirm/category pipeline, so robustness across seeds is not established.
