# Uncertainty and Calibration

## Implemented methods

- Maximum softmax probability (`confidence`) is computed per turn in the severity ensemble.
- Five model seeds are averaged arithmetically. Mean per-class probability variance estimates epistemic disagreement; mean member predictive entropy estimates aleatoric ambiguity.
- Temperature scaling learns one positive scalar on validation ensemble logits by minimizing cross-entropy, then evaluates held-out test probabilities.
- ECE and negative log-likelihood are implemented for the severity prototype. This package additionally reconstructs Brier scores from saved predictions.
- CRADLE abstract alignment computes binary entropy, confidence, and margin from each single model probability and temporal summaries of those values. This is probability-derived uncertainty, not ensemble disagreement.
- The CRADLE uncertainty-augmented policy failed its validation improvement rule and was not selected.

No confidence-margin method was found in the original severity ensemble. No reliability plot, Brier score, selective prediction, or calibrated final CRADLE Confirm artifact was present before this audit. Thresholds for prototype high-uncertainty flags use the dataset-wide 90th percentile in `temporal_features.py`; that thresholding is analysis-oriented and should not be treated as a frozen validation-only deployment threshold.

## Reconstructed metrics

| model | dataset | split | calibration_status | n | ece_10_bin | brier_score | negative_log_likelihood | average_precision | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Confirm | CRADLE-Dialogue | clinician test | uncalibrated | 4527 | 0.0737363065339607 | 0.07380288840183835 | 0.5267079769021714 | 0.6430111066527171 | cradle_dialogue/predictions/single_task_confirm/seed_1_test_predictions.csv |
| B_pos_weight_x2 Confirm | CRADLE-Dialogue | clinician test | uncalibrated | 4527 | 0.06848875711680849 | 0.06820986215722238 | 0.6203209084025687 | 0.6518664327800678 | cradle_dialogue/predictions/single_task_confirm_high_recall/B_pos_weight_x2/seed_1_test_predictions.csv |
| Five-seed severity ensemble | constructed severity dataset | test | uncalibrated | 2235 | 0.00608003479124232 | 0.018194950348577998 | 0.03152784192081984 | NA | outputs/ensemble_uncertainty_test.csv |
| Five-seed severity ensemble | constructed severity dataset | test | temperature-scaled | 2235 | 0.005263489038540089 | 0.018171773057742348 | 0.03129592125319518 | NA | outputs/ensemble_uncertainty_test_calibrated.csv |

The CRADLE rows are uncalibrated clinician-test diagnostics only. Comparing original and high-recall Confirm calibration is descriptive because the high-recall variant intentionally changes loss and threshold. The severity calibrated/uncalibrated files lack stable row identifiers; their row order and labels match, but checkpoint/log provenance should be confirmed before publication.
