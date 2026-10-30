# Figure Data Limitations

- Figure 6 shows the final Confirm model's uncalibrated reliability only. No calibrated final Confirm probabilities exist. Creating the requested calibrated comparison requires fitting calibration on validation logits/probabilities and applying it once to test; retraining the encoder is not required, but saved logits would be preferable.
- Figure 8 compares only the consistent three-class severity encoder benchmark. It must not be interpreted as an Alert/Confirm encoder ablation.
- Figure 9 withholds raw text and uses probability trajectories only. The probability-derived CRADLE uncertainty fields are omitted because they were not selected into the final policy; adding them is possible from `abstract_aligned_features` but risks implying final-system use.
- Figure 11 mixes clearly labeled metrics because no single factorial ablation covers all components. There is no valid end-to-end calibration or five-seed CRADLE ablation.
- No confidence intervals are available for encoder and model classification figures across CRADLE seeds because only seed 1 was run.
