# Implementation-Faithful Equations

1. **Binary logit and probability.** (z=h_\theta(x)), (p=\sigma(z)=1/(1+e^{-z})). Used for CRADLE Alert/Confirm/category heads. Sources: `train_cradle_single_task.py:141-154,222-266`; `train_cradle_single_task_confirm_high_recall.py:137-160`.

2. **Weighted BCE.** (L=-[w_+y\log p+(1-y)\log(1-p)]), masked to evaluable user turns. Source: `train_cradle_single_task.py:518-582`.

3. **Confirm positive weight.** (w_{base}=N_-/N_+); for B, (w_B=2w_{base}=23.158823529411766). Source: `train_cradle_single_task_confirm_high_recall.py:188-203,357-376,560-565`.

4. **Multiclass softmax.** (p_k=\exp(z_k)/\sum_j\exp(z_j)). Source: `ensemble_uncertainty.py` prediction loop.

5. **Temperature scaling.** (p_k^{(T)}=\mathrm{softmax}(z_k/T)), with (T>0) fit by validation cross-entropy. Source: `temperature_scaling.py:learn_temperature`.

6. **Ensemble mean.** (\bar p_k=M^{-1}\sum_{m=1}^M p_{mk}), (M=5). Source: `ensemble_uncertainty.py:main`.

7. **Ensemble variance.** (u_{epi}=K^{-1}\sum_k \mathrm{Var}_m(p_{mk})). Source: `ensemble_uncertainty.py:main`.

8. **Predictive entropy.** (H(p)=-\sum_k p_k\log p_k); the severity prototype averages member entropy for its aleatoric proxy. Source: `ensemble_uncertainty.py:entropy,main`.

9. **Binary confidence margin.** (m=|2p-1|); confidence (=\max(p,1-p)). Source: `align_pipeline_with_abstract.py` probability-derived uncertainty helpers.

10. **Rolling mean.** (\bar p_t^{(w)}=|W_t|^{-1}\sum_{i\in W_t}p_i), where (W_t) contains the current and up to (w-1) preceding user turns. Source: `build_cradle_progression_features_high_recall_confirm.py`.

11. **Probability change.** (\Delta p_t=p_t-p_{t-1}). Source: same progression builder.

12. **Trajectory slope.** OLS slope of observed (p_1,\ldots,p_t) against turn index. Source: `temporal_features.py:slope_so_far` (AnnoMI prototype only).

13. **Baseline deviation.** (d_t=p_t-\bar p_{1:\min(3,n)}). Source: `temporal_features.py:add_conversation_features`; note its full-group implementation is not strictly online for earliest rows.

14. **Learned policy.** (\hat\ell=f_\phi(X_t)\), where (f_\phi) is the validation-selected classifier and (\ell\in\{0,1,2,3\}). Source: `train_cradle_learned_policy.py`.

15. **Safety constraint.** (ell_{final}=\max(\hat\ell,\ell_{guardrail})). Source: `train_cradle_safety_constrained_policy_v2.py:468-490`.

16. **Unsafe continuation.** (UC=\sum_i I(H_i\land \ell_i=0)/\sum_i I(H_i)), (H_i) = Confirm or any category target. Source: `train_cradle_safety_constrained_policy_v2.py:evaluate_policy`.

17. **Over-escalation.** (OE=\sum_i I(N_i\land \ell_i\ge2)/\sum_i I(N_i)), (N_i) = no Alert, Confirm, or category target. Source: same evaluator.
