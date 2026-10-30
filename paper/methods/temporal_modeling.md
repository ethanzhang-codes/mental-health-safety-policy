# Temporal Modeling

| variable | definition | window | first_turn_handling | turn_population | causal | policy_use |
| --- | --- | --- | --- | --- | --- | --- |
| prev_alert_prob | previous user-turn Alert probability | 1 turn | first turn NaN | user only | yes | policy feature |
| alert_prob_change_from_prev | alert_prob - prev_alert_prob | 1 turn | first turn NaN | user only | yes | policy feature |
| alert_rolling_mean_2 | mean of current/previous Alert probabilities | 2 | min_periods=1 | user only | yes | policy feature |
| alert_rolling_mean_3 | mean of up to 3 current/prior Alert probabilities | 3 | min_periods=1 | user only | yes | policy feature |
| alert_rolling_max_3 | max of up to 3 current/prior Alert probabilities | 3 | min_periods=1 | user only | yes | policy feature |
| alert_cumulative_max | maximum Alert probability through current turn | all past | current value | user only | yes | policy feature |
| confirm counterparts | same operations applied to Confirm probabilities | 1/2/3/all | same | user only | yes | policy feature |
| per-category previous/change/cumulative | previous, delta, cumulative max per category | 1/all | NaN/current | user only | yes | policy feature |
| alert_before_confirm_flag | whether first thresholded Alert precedes first thresholded Confirm so far | all past | 0 until observed | user only | yes | policy feature |
| risk_accumulation_score | implementation-defined combination of cumulative Alert/Confirm/category evidence | all past | available from current values | user only | yes | policy feature |
| risk_escalation_score | implementation-defined combination of rises/order/current evidence | current + past | available from current values | user only | yes | policy feature |
| risk_volatility_score | variability/change summary over observed probabilities | current + past | available from observed values | user only | yes | policy feature |
| previous_high_prob | previous severity high-risk probability (AnnoMI prototype) | 1 | NaN | all supplied dialogue turns | yes | prototype policy |
| trajectory_slope_so_far | OLS slope over high-risk probabilities through current turn | all past | 0 with <2 turns | all supplied dialogue turns | yes | prototype policy |
| baseline_deviation | current high probability minus mean of first up-to-3 turns | baseline 3 | defined from available prefix | all supplied dialogue turns | qualified* | prototype policy |
| change_point_flag | first PELT change point; strongest-shift fallback if unavailable | min 6 | 0 when insufficient | all supplied dialogue turns | qualified* | prototype policy |

The final CRADLE progression builder first filters to user turns, sorts by dialogue and turn, and computes current/past-only Alert, Confirm, and category summaries. Leakage columns such as future Confirm distance and final progression stage are excluded. The generated feature audit reports zero missing merged predictions and expected row counts.

Temporal Alert is a separate learned experiment over frozen turn embeddings. History window 8 outperformed single-turn Alert on validation AUROC/AUPRC and clinician-test AUROC/AUPRC. Temporal Confirm performed worse than single-turn Confirm and was not selected. Therefore history improved the evaluated Alert task, not Confirm generally.

The earlier AnnoMI `temporal_features.py` includes severity trajectory and change-point features. PELT is used when `ruptures` is installed and at least six values exist; otherwise a strongest-before/after-mean-shift fallback is used. Two qualifications matter: the baseline is defined from the first up-to-three values of the full supplied group, and PELT is run on the full supplied trajectory. Although features are written per turn, those implementations can expose later values to early rows and should not be represented as strictly online-causal without correction. They are prototype features, not the final CRADLE progression implementation.

All final CRADLE features can be reproduced from saved Alert, Confirm, and category prediction CSVs with `build_cradle_progression_features_high_recall_confirm.py`.
