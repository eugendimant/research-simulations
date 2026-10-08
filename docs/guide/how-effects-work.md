# How effects work

A simulated dataset needs a rule for how conditions differ. This page explains the two kinds of rules the tool uses, how closely the data follow them, and how to see what was built in.

## Effects you specify

Open **Advanced Settings** and add an expected effect size. You choose:

- the dependent variable (a scale from your survey),
- the factor and the two conditions to contrast (which one should score higher),
- the Cohen's d and the direction.

The generator shifts the response tendency of participants in the two conditions, half the gap up and half down. That shift is **calibrated to the scale you are measuring**. A given shift produces a different standardized difference depending on how many items are averaged: items share person-level variance, so the mean of several items has less noise than one item. Without a correction, the d on a ten-item scale mean would be about twice the d on a single item. The calibration makes the requested d the target for the **scale mean** (or for the item, on a single-item scale).

When the design has more than one scale, the effect is instead added to the generated answers of the scale it targets, sized from that scale's own within-condition standard deviation. Answers stay whole numbers within the scale range, and the realised d keeps its natural sampling variation (it is not forced to the request). Without this, the cross-scale correlation structure added noise that cut the realised d to about 0.2-0.5 of the request (0.38 with two scales in the checks). With it, the average over 8 runs is 0.97 of the request with two scales and 1.03 with eight (N = 1,000 and 600, d = 0.5); the single-scale case is unchanged.

In a factorial design give each factor its own `factor` name: effects on different factors add up (both main effects of a 2x2 design are recovered), while effects on one factor, for example two treatments against one control, are averaged.

What to expect:

| Question | Answer |
|---|---|
| Which variable has the requested d? | The scale mean (`<Scale>_mean`, which is in `Simulation_Diagnostics.csv`). Individual items show a smaller d, as in real data, because each item carries its own noise. |
| How close does a run get? | Close to the requested value on average, with ordinary sampling variation. One standard error of d is about 0.14 at 100 participants per group and 0.10 at 200 per group. |
| Does it depend on the scale? | The calibration accounts for the number of items, the number of response options (including two- and three-point scales) and very wide scales such as 0-100 sliders. In the 18 cases checked, the average observed d was within 7% of the request (table below). |
| Can other settings move it? | A little. Variables that most people rate near the top of the scale, and willingness-to-pay style variables with a wide spread, pull it toward zero. Careless and inattentive simulated participants dilute it a little more. Other scales in the design do not. |

### How close the calibration gets

Each cell is the observed d divided by the requested d (1.00 is exact), averaged over 12 independent runs of 1,200 participants each with a requested d of 0.5. These runs used random seeds (501-512) that were not used to fit the calibration. One run on its own varies around the average (a standard deviation of 0.1 to 0.15 in these units at this sample size) because of ordinary sampling error.

| Scale | 1 item | 3 items | 4-6 items | 8 items | 12 items | 20 items |
|---|---|---|---|---|---|---|
| 2-point (0/1) | 0.98 | 1.03 | 1.02 | | | |
| 3-point | 0.97 | 0.98 | 0.96 | | | |
| 5-point | | | 0.94 | 1.00 | | |
| 7-point | 1.07 | 0.97 | 0.99 | 1.02 | 0.98 | 0.99 |
| 11-point (0-10) | | | 0.99 | 1.02 | | |
| 0-100 slider | 1.07 | | | 1.02 | | |

The 4-6 item column holds 4 items for the 5-point and 11-point rows, 5 for the 7-point row and 6 for the 2-point and 3-point rows. Averaged over the 18 cells the absolute error of the cell means is 2.8% (worst cell 7%).

**How a block of three or more items gets its effect (version 1.3.0.6).** A scale of one or two items has the shift built into the response generator, calibrated as described above. A scale of three or more items is first generated and brought to its target reliability (Cronbach's alpha) and item distribution, and only then moved by the requested effect, in units of its own within-condition SD, exactly as is done when the design has several scales. Before this, the step that lowers an overly high inter-item correlation to the target alpha added item noise after the shift was in, and the 4-item 5-point scale recovered 0.88 to 0.92 of the request (1.07 with that step switched off). The step is now aimed slightly lower so that, once the effect has been added, the block still ends on the target alpha. The shift is applied with one randomised-rounding draw per respondent shared by all items, so answers stay integers without breaking the item correlation. Effects the tool infers from the condition names are built in the same way.

For comparison, `main` at version 1.3.0.3 (without this calibration, same cells and seeds) gave 1.12 at 8 items, 1.22 at 12 and 1.29 at 20 on a 7-point scale, 0.75, -0.04 and 0.67 on 2-point scales with 1, 3 and 6 items (the three-item binary scale lost its effect entirely, because the straight-line passes rewrote honest answers), and 0.60 on a three-item 3-point scale. Averaged over the 18 cells the absolute error of the cell means was 19.8% on `main`, 3.5% at version 1.3.0.5 and 2.8% now (worst cell 104%, 9%, 7%). Scales with more than 30 items use the 30-item correction. Effects on scales with reverse-keyed items, or with many careless responders, are attenuated further (see below), by about 20% when two of four items are reversed and by about a third when all four are (8-run checks, d = 0.5). Other scales in the design do not attenuate an effect. The table was measured on a single, plainly keyed scale; the multi-scale route above was checked against the same targets (0.97 and 1.03 for two and eight scales).

## Effects the tool infers

If you do not specify an effect for a variable, the tool reads the condition names and the study description. A "gain" frame versus a "loss" frame, "AI" versus "human", "high" versus "low", a named paradigm with a published estimate, and similar patterns each get a literature-informed difference, so that conditions do not come out identical.

These inferred effects are **a heuristic, not an effect you chose**:

- They are not tuned to a target d. The observed d can differ from what the wording suggests.
- The total inferred shift is capped, so that keyword combinations cannot produce implausible effects.
- **Published effects are shrunk before they are used.** Published effect sizes are inflated by selective reporting, and large replication projects have found replication effects of roughly half the original (Open Science Collaboration 2015, Camerer et al. 2018, Many Labs, Kvarven et al. 2020). So when a named paradigm or a literature-table match sets an inferred effect, the published d is multiplied by 0.60 and each run then draws its own effect from a between-study distribution (SD 0.15 d where the table reports none), so two runs differ the way two labs do. The result never falls below 35% of the published d. The 0.60 is **recalled from that literature, not checked against the papers**, and is labelled that way in the audit tables. It applies to the paradigm anchor, the literature-table fallback and (since version 1.3.0.6) the generic keyword effects (valence, "high" versus "low", "gain" versus "loss", domain keywords). Those keyword magnitudes were set by hand from original findings and gave gaps of d 0.5 to 0.9; replicated effects of such manipulations on attitudes are mostly d 0.2 to 0.5, so they now give 0.3 to 0.55. The intergroup and economic-game effects are not shrunk, and neither is any effect you specify. An entry that is also marked as unchecked gets the stronger of its tier weight and the 0.60, not both multiplied. Every inferred effect is applied to the finished item responses in units of the scale's own SD, like an effect you specify, so the contrast listed in `effect_sizes_applied` is the d the scale mean shows (a fallback match used to deliver only about half of it).

  Realised d on a 4-item, 7-point scale mean (12 runs of 800 participants, version 1.3.0.5 then 1.3.0.6; first condition minus second): gain frame vs loss frame +0.14 / +0.14; high vs low quality +0.58 / +0.40; positive vs negative review +0.25 / +0.25 (paradigm anchor, unchanged); AI vs human recommender -0.12 / -0.06; anthropomorphic vs mechanical robot +0.10 / +0.15; fair vs unfair procedure +0.41 / +0.46; scarcity vs abundant -0.04 / -0.03; mortality salience vs control +0.08 / +0.10; self-affirmation vs control +0.15 / +0.15; "Treatment" vs "Control" +0.24 / +0.15. A match from the literature table now delivers the d it names (a content-matched fallback returned 0.43 of it with one item and 0.23 with four; now 0.98 and 0.82), and the keyword magnitudes fall into the 0.15 to 0.45 band of replicated effects. The sampling error of each figure is about 0.03.
- They apply only to variables you did not specify an effect for. Where you specify an effect for a variable, nothing inferred from the condition names is added on top of it as a mean shift. For conditions named in any effect you specify, the name-based changes to response style are switched off as well.

To remove them, untick **Also infer small differences from the condition names** in Advanced Settings. Then only the effects you specify are built in, and every other contrast is a true null apart from sampling noise. This is the setting to use when you want to check a false-positive rate, or when you do not want the tool to guess.

## Seeing what was built in

Every run writes the following into `Metadata.json`:

- `effect_sizes_configured`: the effects you specified.
- `effect_sizes_applied`: for each variable and pair of conditions, whether the contrast came from your specification (`user`), from the name-based heuristic (`inferred`), or from nothing (`none`), the intended d where one exists, and the d observed in this sample. For inferred contrasts that came from the literature it also gives `published_d`, the `shrinkage_factor` and the `applied_d` that sized the contrast, and `inferred_effect_policy` states the policy (factor, heterogeneity SD, evidence tier). Each contrast is `condition_1` minus `condition_2`, in the order of your conditions, so a contrast listed as Control then Treatment shows a negative d when Treatment scores higher. The summary report shows the same numbers oriented as high level minus low level.
- `effect_sizes_observed`: the observed Cohen's d for every item and scale mean, for every pair of conditions.
- `effect_sizes_applied.specs`: one row per effect you specified, with whether it reached anything (`matched`, `status`: `applied`, `one_side_only`, `levels_not_found` or `variable_not_found`). An effect that matches no variable or condition is also listed in `generation_warnings` and in the quality notes, so a dropped effect is never silent. `applied_after_generation` lists the scales whose effect was added to the finished answers (more than one scale in the design).

`User_Study_Summary.md` shows the same information as tables.

## Study context versus condition names

Study-level context (for example "this is a political study") nudges every condition's response style in the same way and does not create differences between conditions. Only condition names create differences, and only where you have not specified an effect.

## Economic-game outcomes

Outcomes that look like dictator, trust, ultimatum or public-goods games are generated by a behavioral-economics model, which replaces the numeric generator's values for those columns. The model gives the distribution its realistic shape (for example, clusters at zero and at an even split). If you specify an effect for such an outcome, it is restored after the model runs: each condition's mean is moved to its target (plus or minus half the requested contrast, in standard deviations of the outcome), with values rounded and clipped to the scale range. The observed d is therefore slightly below the request when many answers sit at the scale limits. Check `effect_sizes_observed` for these outcomes. Game outcomes without a specified effect keep the model's own condition differences.

## Scales with reverse-keyed items

The simulator includes participants who ignore item direction, in line with survey research (roughly 10 to 15% of respondents, more among careless responders). On a scale with reverse-keyed items this attenuates the observed effect on the scale mean, as it does in real data. `<Scale>_mean` is computed after reverse-scoring, so it measures the effect as a researcher would analyze it.
