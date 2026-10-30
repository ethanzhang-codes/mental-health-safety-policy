# Limitations

- CRADLE train/validation differ from the clinician test distribution; large validation-to-test gaps are visible.
- The final CRADLE pipeline is single-seed.
- Policy targets are rules derived from dataset labels, not independently validated clinical action labels.
- Final unsafe continuation remains 22.98% on eligible clinician-test turns.
- Category performance is weak on clinician test, especially rare categories.
- Confirm is current-turn-only; existing error analysis suggests many misses need context.
- Final CRADLE probabilities are not temperature calibrated.
- Probability-derived entropy/margin is not equivalent to Bayesian or ensemble uncertainty; the uncertainty policy ablation was rejected.
- Hand-written guardrails can encode brittle thresholds and unmeasured bias.
- Temporal summaries compress history and may lose wording, speaker, and long-range relationships.
- AnnoMI pseudo-label metrics cannot support clinical claims or direct comparison with CRADLE.
- Dataset licensing, demographic representativeness, privacy procedures, IRB status, and external validity are unresolved.
- Multiple exploratory test evaluations limit strong confirmatory interpretation.
