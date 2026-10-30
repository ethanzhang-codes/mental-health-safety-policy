# Project Walkthrough: A Safety Layer for a Mental-Health Chatbot

This project studies a **safety decision layer** for a mental-health chatbot. It is not a therapist, a diagnostic system, or a replacement for emergency or clinical services. Its job is narrower: notice possible risk, use the conversation so far to judge how the risk is changing, and choose a safer response level.

The fixed research goal is to combine four kinds of evidence:

- what the user says now;
- what happened earlier in the conversation;
- how confident or uncertain the models are; and
- explicit safety rules that can prevent a learned policy from continuing when warning signs are strong.

The project then evaluates the tradeoff between missing risk and escalating too often.

## 1. Overall research problem

A chatbot that considers only the latest message can miss a crisis that develops gradually. For example, “I have had a terrible week” may not be enough to trigger escalation by itself. If later messages become more specific or urgent, however, the sequence can be much more concerning.

The research question is: **Can a chatbot safety layer combine current-turn predictions, conversation history, uncertainty, and guardrails to choose a proportionate action?** The desired behavior is not to escalate every difficult conversation. It is to continue ordinary supportive dialogue when appropriate, provide resources when concern rises, and involve a human when the evidence is strong enough.

This is an experimental safety framework. Results on research datasets do not establish clinical effectiveness or justify real-world autonomous deployment.

## 2. Role of AnnoMI in the initial prototyping stage

AnnoMI contains motivational-interviewing conversations. It was useful for building the first end-to-end prototype because it supplied realistic multi-turn dialogue structure: conversation identifiers, speakers, turn order, and text.

The prototype used AnnoMI to test whether the engineering pieces could connect: single-turn risk scores, ensemble uncertainty, temporal features, an escalation policy, rationales, and safety-centered reports.

AnnoMI does **not** provide the clinical risk and escalation labels needed for the intended safety evaluation. Some early reports therefore compare predictions with pseudo-labels derived from those same predictions. Metrics from that setup are sanity checks only. For example, a perfect score against a model's own pseudo-label is not evidence of perfect clinical prediction.

In short, AnnoMI answered, “Can we assemble and exercise the pipeline?” It could not reliably answer, “Is this pipeline clinically safe?”

## 3. Role of CRADLE-Dialogue in the upgraded evaluation stage

CRADLE-Dialogue is the upgraded evaluation source because it includes explicit annotations for crisis-related dialogue events. The project converts those annotations into turn-level targets while preserving the original train, validation, and clinician-test splits.

The main targets are **Alert**, **Confirm**, and seven Confirm-linked crisis categories. Alert represents an earlier warning signal. Confirm represents more explicit evidence that a crisis category applies. The categories cover the kinds of risk represented in CRADLE, including suicide/self-harm and interpersonal-harm categories.

CRADLE also makes temporal evaluation possible: the system can ask whether Alert appears before Confirm and whether risk evidence accumulates across user turns. Features are built causally, using only the current and previous turns. Future annotations may define an evaluation target, but they are forbidden as model input.

CRADLE does not supply the project's separate Low/Medium/High severity labels. It therefore strengthens the Alert, Confirm, category, progression, and policy experiments, but it does not by itself complete every dimension of the proposed framework.

## 4. Single-turn Alert and Confirm prediction

The first prediction question is local: **What does this user turn say on its own?**

- **Alert prediction** estimates whether the current turn contains an early crisis warning.
- **Confirm prediction** estimates whether the current turn contains explicit crisis-confirming evidence.

Imagine a user first says, “Things are getting hard to manage,” and later gives a clear statement of immediate danger. The earlier message may receive a higher Alert probability, while the later one may receive a higher Confirm probability.

These are probabilities, not diagnoses. A decision threshold turns each probability into a yes/no prediction. Because missed Confirm cases are safety-critical, the project tested higher-recall Confirm variants and selected `B_pos_weight_x2` using validation data only. Its selected threshold is `0.011`. On validation, recall increased at the cost of more false positives; test results are reporting-only and were not used to select the model.

The project has also tested temporal Alert modeling, where representations of earlier turns supplement the current turn. This is distinct from the basic single-turn task and supports the progression stage described below.

## 5. Probability calibration and uncertainty estimation

A probability should mean more than “the model produced a large number.” If predictions assigned about 80% confidence are correct only half the time, the model is overconfident. **Calibration** measures and corrects this mismatch.

The prototype includes temperature scaling. It learns one scaling value on validation logits and applies it before computing test probabilities. Calibration is assessed with negative log-likelihood and expected calibration error, alongside accuracy and average confidence.

The prototype also uses an ensemble of models trained with different random seeds. Their average gives the final severity probability. Their disagreement is used as an estimate of **epistemic uncertainty**—roughly, “the models do not agree because their learned knowledge differs.” Average predictive entropy is used as an estimate of **aleatoric uncertainty**—roughly, “the input itself is ambiguous.”

For example, five models that strongly disagree about a vague message should produce a high-uncertainty flag. The policy can then act more cautiously instead of treating the average score as certain.

Calibration and ensemble uncertainty are implemented in the initial severity prototype. They are not yet fully integrated and validated across every CRADLE Alert, Confirm, category, progression, and policy component. Uncertainty quality therefore remains an evaluation and development target, not a solved result.

## 6. Temporal risk features derived from conversation history

The pipeline records how risk evidence changes over user turns. It computes causal features such as:

- the previous Alert or Confirm probability;
- change from the previous user turn;
- rolling means and recent maximums;
- the highest probability seen so far;
- whether risk is rising;
- whether Alert occurred before Confirm;
- time between the first Alert and first Confirm;
- category changes; and
- accumulation, escalation, and volatility scores.

For example, probabilities of `0.10`, `0.35`, and `0.70` form a different pattern from a single isolated `0.70`. The rising sequence provides evidence of progression even though both examples end at the same current score.

The feature builder explicitly excludes information from future turns, such as “turns until Confirm” or a final progression-stage label. This leakage protection matters because a deployed system cannot know what a user will say next.

## 7. Multi-dimensional risk assessment: severity, category, and progression

The intended framework considers three related but different questions:

1. **Severity:** How serious does the current text appear—Low, Medium, or High risk?
2. **Category:** What kind of crisis evidence is present, such as suicide/self-harm or interpersonal harm?
3. **Progression:** Is the evidence stable, rising, volatile, accumulating, or moving from Alert to Confirm?

These dimensions should not be collapsed conceptually. Two messages can have similar severity scores but refer to different hazards. Likewise, a moderate current score may deserve more attention when it follows a sustained upward trend.

The current repository contains a separate severity benchmark and CRADLE models/features for Alert, Confirm, category, and progression. It does **not** yet contain a finished, jointly trained, clinically validated model that unifies all three dimensions. The current policy experiments combine available prediction and progression features, but the full multi-dimensional modeling goal remains under development.

## 8. Graded escalation policy

The CRADLE policy uses four internal levels. They fit into three broad user-facing response families:

| Internal level | Response family | Plain-language meaning |
| --- | --- | --- |
| Level 0 | Continue conversation | Risk signals are low and stable, so continue normal supportive dialogue. |
| Level 1 | Provide support resources | Offer supportive resources or a light safety check. |
| Level 2 | Defer to human support | Recommend or require human support when evidence is stronger. |
| Level 3 | Defer urgently | Use urgent escalation when Confirm and/or high-risk category evidence is especially strong. |

This grading avoids a false choice between “do nothing” and “trigger an emergency response.” For example, a moderate Alert probability might lead to resources, while strong Confirm evidence paired with suicide/self-harm evidence can set an urgent minimum level.

The repository compares a rule-based policy, an unconstrained learned policy, and a learned policy with safety constraints. Current test results illustrate a tradeoff rather than a victory: the high-recall Confirm version reduced unsafe continuation for the safety-constrained policy from about 29.8% to 23.0%, while over-escalation increased slightly from about 11.9% to 12.2%. The rule-based policy had lower unsafe continuation in that experiment but a much higher over-escalation burden. These dataset-specific results are not clinical safety claims.

## 9. Safety guardrails

A learned policy may favor the common “continue” class and miss rare dangerous cases. Guardrails place a minimum escalation level on selected patterns, even if the learned policy proposes a lower action.

Current guardrail signals include high Confirm probability, elevated suicide/self-harm category probabilities, accumulated Alert risk, and combinations of moderate evidence. Several guardrail variants are selected using validation data, not test data.

Guardrails reduce some unsafe continuations but can increase false alarms. They are therefore audited with both unsafe-continuation and over-escalation measures. The next stated implementation step is to inspect remaining high-risk Level 0 cases and adjust rules only when repeated missed-case patterns justify a change.

Guardrails are a defense layer, not proof of safety. They remain dependent on upstream models: a rule cannot react to evidence that every predictor misses.

## 10. Interpretable rationales

Each policy output includes a short rationale tied to active signals. Examples include:

- “Continue conversation because Alert, Confirm, and category risk estimates are low and stable.”
- “Offer supportive resources because Alert probability is elevated.”
- “Defer to human support because Confirm follows earlier Alert signals.”

These rationales help a researcher or reviewer see why a level was chosen. They also make failure audits easier: a reviewer can compare the stated trigger with the probabilities and history features.

The rationales explain the policy's rules or model features; they do not explain a person's mental state, provide a diagnosis, or guarantee that the decision is correct. Feature importance and template explanations are useful forms of traceability, but they are not complete causal explanations of a neural model.

## 11. Safety-centered evaluation metrics

Ordinary accuracy is not enough because different errors have different safety costs.

**Unsafe continuation** is the fraction of annotated high-risk turns for which the policy selects Level 0, meaning it continues without added support or human deferral. In some older prototype reports, Level 1 is also grouped as insufficient for high risk; metric definitions must therefore be stated with every result. Lower is better.

**Over-escalation** is the fraction of no-risk or low-risk turns sent to a high escalation level. Lower is better, but reducing it by ignoring genuine risk would be unsafe. It must be read together with unsafe continuation.

**Confirm recall** asks: of all true Confirm turns, how many did the Confirm model detect? High recall matters because a false negative can hide explicit crisis evidence. Precision must still be reported because a very low threshold can create many false alarms.

**Calibration** asks whether confidence matches observed correctness. Useful measures include expected calibration error, negative log-likelihood, reliability plots, and class-specific calibration—especially for rare safety-critical classes.

**Uncertainty quality** asks whether uncertainty is actually high on errors, unfamiliar inputs, ambiguous cases, and distribution shifts. Useful tests include error-detection AUROC/AUPRC, selective-risk or risk-coverage curves, uncertainty calibration, and out-of-distribution evaluation. Merely computing entropy or ensemble variance is not enough.

Other useful measures include precision and recall by category, time to first escalation, escalation-level distribution, urgent-escalation precision, human-deferral precision, and autonomous coverage. Results should be split by dataset and subgroup where possible, with confidence intervals and repeated seeds before drawing broad conclusions.

## 12. A 30-second explanation of the project

This project builds a safety layer for a mental-health chatbot. It first looks for an early warning, called Alert, and more explicit crisis evidence, called Confirm. It also tracks the possible crisis category and how signals change across the conversation. Calibrated probabilities, uncertainty estimates, and safety guardrails then help a policy choose whether to continue, offer resources, or defer to a human. The key evaluation asks whether the system avoids unsafe continuation without escalating ordinary conversations too often. It is a research prototype, not a clinically validated system.

## 13. Complete pipeline diagram

```text
User and chatbot conversation
            |
            v
Current user turn + causal history (no future turns)
            |
            +---------------------------+
            |                           |
            v                           v
 Single-turn evidence              History encoder /
 Alert, Confirm, category          temporal Alert model
 probabilities                     (experimental)
            |                           |
            +-------------+-------------+
                          |
                          v
              Probability calibration
              + uncertainty estimates
              (integration still incomplete)
                          |
                          v
              Causal progression features
      previous values, rolling trends, Alert -> Confirm,
          accumulation, escalation, volatility, categories
                          |
                          v
        Multi-dimensional evidence for policy decisions
          severity | category | progression | uncertainty
          (full unified model is still under development)
                          |
                          v
              Learned or rule-based policy
                          |
                          v
                   Safety guardrails
             enforce minimum response levels
                          |
                          v
              Graded action + rationale
       continue | resources | human/urgent deferral
                          |
                          v
             Safety-centered evaluation
 unsafe continuation | over-escalation | Confirm recall
        calibration | uncertainty quality | timing
```

## 14. What is complete and what is still under development?

Here, “complete” means implemented and exercised in the repository. It does **not** mean clinically validated, production-ready, or finished research.

### Implemented and exercised

- AnnoMI end-to-end prototype for severity prediction, uncertainty, temporal features, escalation, rationales, and reporting.
- Explicit documentation in the AnnoMI evaluation that pseudo-label results are sanity checks, not clinical performance.
- CRADLE preprocessing, fixed splits, target construction, leakage checks, and audits.
- CRADLE single-turn Alert and Confirm training and evaluation.
- Validation-only selection of the `B_pos_weight_x2` high-recall Confirm candidate.
- A seven-label, single-turn CRADLE category model, with known weak clinician-test performance for some categories.
- Temporal Alert experiments using frozen turn embeddings.
- Causal CRADLE progression features combining Alert, Confirm, and category predictions.
- Rule-based, learned, and safety-constrained graded policy experiments.
- Guardrail variants, per-row policy rationales, safety metrics, and false-negative audits.
- An isolation experiment showing how the selected high-recall Confirm candidate changes downstream policy results.

### Still under development or requiring stronger validation

- A single unified model of severity, category, and progression. The current pieces are separate or combined at the feature/policy level.
- Full integration of calibrated probabilities and uncertainty into the upgraded CRADLE pipeline.
- Thorough evaluation of whether uncertainty reliably identifies errors, unfamiliar inputs, and dataset shift.
- Stronger and more consistent category performance, especially for rare categories.
- Reduction of remaining high-risk Level 0 cases without unacceptable over-escalation.
- Guardrail tuning based on structured audits of repeated failure patterns.
- Multi-seed, confidence-interval, subgroup, robustness, and out-of-distribution evaluation across the full pipeline.
- External and prospective validation with appropriate clinical and safety oversight.
- Human-factors testing of resources, deferral behavior, rationales, and failure handling.
- Production safeguards, monitoring, privacy review, and emergency-service localization.

The honest current conclusion is that the repository demonstrates a substantial experimental pipeline and exposes its safety tradeoffs. It does not yet demonstrate a complete multi-dimensional clinical risk model or a chatbot that is safe for autonomous real-world use.
