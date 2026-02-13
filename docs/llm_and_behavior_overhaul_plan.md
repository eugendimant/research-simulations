# LLM + Behavior Overhaul Plan (v1.0.6.9)

## A. 3 Iterations for LLM reliability/admin correctness

1. **Iteration 1 (hotfix):** fix engine init crash (`os` import) so LLM generator can initialize.
2. **Iteration 2 (pipeline ordering):** move Google AI Studio providers to top of built-in chain.
3. **Iteration 3 (visibility):** expose stronger quality/fallback counters in LLM stats and keep admin diagnostics consistent with actual call attempts.

## B. 6 Iterations (1+5) for open-ended quality

### Baseline plan
1. Ensure prompts carry explicit question+condition context.
2. Add response quality gate before accepting pooled outputs.
3. Reject generic/gibberish patterns.
4. Enforce lexical diversity minimums for longer answers.
5. Enforce topical relevance token overlap with question/condition.

### Additional 5 refinement iterations
6. Reject known garbage stock fragments seen in bad outputs.
7. Apply the same quality check to pool-drawn responses (not only fresh batches).
8. Track quality rejection counts for diagnostics.
9. Keep anti-AI artifact cleanup but avoid over-normalizing all responses.
10. Add regression tests for fallback-policy and condition integrity after OE processing.

## C. 20-step plan for more human-like DV responses

1. Preserve treatment effects as primary signal.
2. Add participant-level stable latent style (already present; keep explicit weighting).
3. Add bounded drift across longer scales (fatigue toward midpoint when low attention).
4. Add lightweight response inertia (previous-item pull) for low-consistency responders.
5. Add occasional correction of endpoint overuse for high-attention responders.
6. Keep reverse-coded failure probabilities heterogeneous by attention/engagement.
7. Keep acquiescence and extremity as independent style dimensions.
8. Include social-desirability sensitivity by topic domain.
9. Preserve condition × persona interaction heterogeneity.
10. Keep cross-DV coherence via participant response history.
11. Avoid deterministic trajectories across items (seeded stochasticity).
12. Clamp all generated values to valid item bounds.
13. Keep item-level variance realistic (not over-smoothed composites).
14. Keep per-condition means plausible with configured Cohen’s d.
15. Prevent synthetic-perfect reliability (alpha inflation checks remain).
16. Keep response-time/quality coupling plausible.
17. Maintain occasional satisficing behaviors rather than idealized responding.
18. Keep heterogeneity in endpoint use by persona.
19. Keep condition balance and assignment independent from text post-processing.
20. Preserve auditability by logging generation assumptions and versioned changes.

## D. Literature note (Westwood 2025 PNAS)

Attempted to fetch bibliographic metadata from Crossref in this runtime but network/proxy restrictions blocked access. The implementation therefore uses robust survey-method findings already represented in-code (Krosnick, Greenleaf, Meade & Craig, Podsakoff, etc.) and extends them conservatively.
