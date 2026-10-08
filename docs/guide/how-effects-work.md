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
| Does it depend on the scale? | The calibration accounts for the number of items, the number of response options (including two- and three-point scales) and very wide scales such as 0-100 sliders. In the 18 cases checked, the average observed d was within 12% of the request, and within 8% in 16 of them (table below). |
| Can other settings move it? | A little. Variables that most people rate near the top of the scale, and willingness-to-pay style variables with a wide spread, pull it toward zero. Careless and inattentive simulated participants dilute it a little more. Other scales in the design do not. |

### How close the calibration gets

Each cell is the observed d divided by the requested d (1.00 is exact), averaged over 12 independent runs of 1,200 participants each with a requested d of 0.5. These runs used random seeds (401-412) that were not used to fit the calibration. One run on its own varies around the average (a standard deviation of 0.1 to 0.15 in these units at this sample size) because of ordinary sampling error.

| Scale | 1 item | 3 items | 4-6 items | 8 items | 12 items | 20 items |
|---|---|---|---|---|---|---|
| 2-point (0/1) | 0.97 | 0.92 | 0.97 | | | |
| 3-point | 0.98 | 1.01 | 0.93 | | | |
| 5-point | | | 0.88 | 0.96 | | |
| 7-point | 1.05 | 0.96 | 0.98 | 0.99 | 1.03 | 1.05 |
| 11-point (0-10) | | | 0.98 | 1.01 | | |
| 0-100 slider | 1.05 | | | 1.01 | | |

The 4-6 item column holds 4 items for the 5-point and 11-point rows, 5 for the 7-point row and 6 for the 2-point and 3-point rows. The weakest cell is the 4-item 5-point scale (0.88). There the step that lowers an overly high inter-item correlation to the target Cronbach's alpha adds item-level noise to the composite; with that step switched off the same cell gives 1.07.

For comparison, `main` at version 1.3.0.3 (without this calibration, same cells and seeds) gave 1.12 at 8 items, 1.22 at 12 and 1.29 at 20 on a 7-point scale, 0.75, -0.04 and 0.67 on 2-point scales with 1, 3 and 6 items (the three-item binary scale lost its effect entirely, because the straight-line passes rewrote honest answers), and 0.60 on a three-item 3-point scale. Averaged over the 18 cells the absolute error of the cell means was 19.8% on `main` and 3.8% here (worst cell 104% against 12%). Scales with more than 30 items use the 30-item correction. Effects on scales with reverse-keyed items, or with many careless responders, are attenuated further (see below), by about 20% when two of four items are reversed and by about a third when all four are (8-run checks, d = 0.5). Other scales in the design do not attenuate an effect. The table was measured on a single, plainly keyed scale; the multi-scale route above was checked against the same targets (0.97 and 1.03 for two and eight scales).

## Effects the tool infers

If you do not specify an effect for a variable, the tool reads the condition names and the study description. A "gain" frame versus a "loss" frame, "AI" versus "human", "high" versus "low", a named paradigm with a published estimate, and similar patterns each get a literature-informed difference, so that conditions do not come out identical.

These inferred effects are **a heuristic, not an effect you chose**:

- They are not tuned to a target d. The observed d can differ from what the wording suggests.
- The total inferred shift is capped, so that keyword combinations cannot produce implausible effects.
- **Published effects are shrunk before they are used.** Published effect sizes are inflated by selective reporting, and large replication projects have found replication effects of roughly half the original (Open Science Collaboration 2015, Camerer et al. 2018, Many Labs, Kvarven et al. 2020). So when a named paradigm or a literature-table match sets an inferred effect, the published d is multiplied by 0.60 and each run then draws its own effect from a between-study distribution (SD 0.15 d where the table reports none), so two runs differ the way two labs do. The result never falls below 35% of the published d. The 0.60 is **recalled from that literature, not checked against the papers**, and is labelled that way in the audit tables. It applies only to the paradigm anchor and the literature-table fallback; the generic keyword effects (valence, "high" versus "low", and so on) and the economic-game baselines are not shrunk, and neither is any effect you specify.
- They apply only to variables you did not specify an effect for. Where you specify an effect for a variable, nothing inferred from the condition names is added on top of it as a mean shift. For conditions named in any effect you specify, the name-based changes to response style are switched off as well.

To remove them, untick **Also infer small differences from the condition names** in Advanced Settings. Then only the effects you specify are built in, and every other contrast is a true null apart from sampling noise. This is the setting to use when you want to check a false-positive rate, or when you do not want the tool to guess.

## Seeing what was built in

Every run writes the following into `Metadata.json`:

- `effect_sizes_configured`: the effects you specified.
- `effect_sizes_applied`: for each variable and pair of conditions, whether the contrast came from your specification (`user`), from the name-based heuristic (`inferred`), or from nothing (`none`), the intended d where one exists, and the d observed in this sample. For inferred contrasts that came from the literature it also gives `published_d`, the `shrinkage_factor` and the `applied_d` that sized the contrast, and `inferred_effect_policy` states the policy (factor, heterogeneity SD, evidence tier). Each contrast is `condition_1` minus `condition_2`, in the order of your conditions, so a contrast listed as Control then Treatment shows a negative d when Treatment scores higher. The summary report shows the same numbers oriented as high level minus low level.
- `effect_sizes_observed`: the observed Cohen's d for every item and scale mean, for every pair of conditions.
- `effect_sizes_applied.specs`: one row per effect you specified, with whether it reached anything (`matched`, `status`: `applied`, `one_side_only`, `levels_not_found` or `variable_not_found`). An effect that matches no variable or condition is also listed in `generation_warnings` and in the quality notes, so a dropped effect is never silent. `applied_after_generation` lists the scales whose effect was added to the finished answers (more than one scale in the design).

`User_Study_Summary.md` shows the same information as tables.

## Within-subjects and mixed designs

When every participant answers every condition (a within-subjects design) or answers several conditions inside a between-subjects group (a mixed design, such as treatment/control x Pre/Post), "the effect" needs a definition that fits the repeated structure.

**What the requested d means: d_av.** An effect you set on a within factor is the mean difference between two conditions divided by the **average of the two conditions' SDs** (d_av; Lakens 2013). That is the quantity that is comparable to a between-subjects d: the same d = 0.5 describes the same shift of the scores in standard-deviation units, whichever design measures it. The paired effect size d_z (mean difference divided by the SD of the differences) is reported next to it. They are linked by `d_z = d_av / sqrt(2 (1 - r))`, where r is the correlation between the two conditions, so the same d_av gives a larger d_z, and a more powerful paired test, when the conditions are more strongly correlated: at r = 0.5 they are equal, at r = 0.7 d_z is 1.29 times d_av.

**The within-person correlation.** The same measure is correlated across conditions through a person-level latent: by default r = 0.5 (attitude measures typically show test-retest and repeated-measures correlations between 0.4 and 0.7; the default is a round value inside that range, recalled from the literature rather than fitted to data). You can set r between 0 and 0.9 on the Design page, or choose an AR(1) structure (`r ** lag`) for time points. r is the correlation of the scores as you will see them in the whole sample. The engine reaches it in three steps: the cross-condition coupling of the generator, a top-up that adds a shared person-level component when the sample falls short, and, when the sample comes out above r because careless responders repeat the same answer, a lowering step that adds condition-specific noise. For single-item measures the measurement error that real single items carry already lowers the correlation, and the same calibration applies.

**How your effects map onto the design.** Each effect names a factor and the two levels to contrast (which one scores higher), exactly as in a between-subjects design:

- On a within factor the contrast is built into the same people: their score in one condition is shifted relative to the other, in units of the measure's own SD, on top of their person-level intercept. A lone contrast splits symmetrically (+d/2 for the higher level, -d/2 for the lower), as in the between-subjects route. Several effects on one factor are combined by least squares (A - B = 0.4 and B - C = 0.4 give A - C = 0.8); effects on different factors add, so both main effects of a 2 x 2 within design are recovered.
- In a mixed design an effect on the between-subjects factor is present at every level of the within factor *except the first, baseline level* (Pre, T1, Baseline, Wave 1), where randomized groups do not differ. To say exactly where it applies, give the effect an `at` (for example `at={"Time": "Post"}`).
- A group x time **interaction** is written as a group effect restricted to one within level (`at`), combined with, or without, a main effect of time. In the app: choose "Group difference at one Time level (interaction)" when you add an expected effect. Through the engine API the same effect goes into `design["simple_effects"]`.
- Effects inferred from the condition names work in within designs too. The reference level (a control, baseline or "pre" label) is the zero point, exactly as for a between-subjects control arm; inferred effects are not added to a variable that has an effect you specified.

**Order, fatigue and attrition.** The order in which each participant saw the conditions (random, a balanced Latin square, every permutation up to five conditions, or one fixed order) is recorded in `Order` and `Position_<Condition>`. A small drift of -0.05 SD per later position (mild fatigue) is added by default; it is recalled from the literature on repeated ratings (practice and fatigue effects are usually small relative to a manipulation), is not fitted to data, and can be switched off or changed. With a fixed order the drift is confounded with the conditions, as it would be in a real study. A participant who drops out loses the conditions presented *after* the last one they finished, so the missing data of a within design have the usual attrition pattern.

**Careless responders.** A person who straight-lines does so in every condition and receives neither the effect nor the correlation shift. Because the effect is a statement about the sample, it is scaled up by the careless share (capped at 1.25) so that the sample-level d_av still lands on your request; careless respondents therefore dilute nothing on average, and the individual-level effect among attentive respondents is a little larger than d.

### How close the within-subjects calibration gets

Observed d_av divided by the requested d_av, averaged over 12 independent runs of 300 participants each (requested d_av = 0.5, requested r = 0.5, counterbalanced with a Latin square, 7-point scales; seeds 101-112 for all cells). The effect was built into the contrast between the first and the last condition.

| Measure | 2 conditions | 4 conditions |
|---|---|---|
| 4 items, alpha 0.80-0.90 | 0.96 | 0.96 |
| 1 item | 0.96 | 0.99 |

One run on its own varies by about 0.04 to 0.06 in d_av at N = 300 (the standard deviation across the 12 seeds), because the contrast is estimated on 300 people. The correlation between conditions came out at 0.49 to 0.51 across the four cells (target 0.5).

### What `Metadata.json` records for a repeated-measures run

- `design`: the type, the within factors and the conditions (label and the suffix used in column names), the order scheme, the within-person correlation requested and how it was reached (`coupling`: the engine's own correlation, the final one, the weights of the top-up and the lowering), the attrition that was applied, the number of straight-liners, and `wide_columns` (for each measure and condition, the item columns and the composite).
- `effect_sizes_applied.specs`: one row per effect you specified, with its scope (`at`), whether it is a `within` or a `between` effect, and `status`; `cell_offsets_d` lists the offset, in SD units, built into every cell.
- `effect_sizes_observed`: for every measure and pair of conditions, the observed d_av, d_z, the paired correlation r and the number of complete pairs. The instructor report checks each effect you requested against the sample, in d_av.

## Study context versus condition names

Study-level context (for example "this is a political study") nudges every condition's response style in the same way and does not create differences between conditions. Only condition names create differences, and only where you have not specified an effect.

## Economic-game outcomes

Outcomes that look like dictator, trust, ultimatum or public-goods games are generated by a behavioral-economics model, which replaces the numeric generator's values for those columns. The model gives the distribution its realistic shape (for example, clusters at zero and at an even split). If you specify an effect for such an outcome, it is restored after the model runs: each condition's mean is moved to its target (plus or minus half the requested contrast, in standard deviations of the outcome), with values rounded and clipped to the scale range. The observed d is therefore slightly below the request when many answers sit at the scale limits. Check `effect_sizes_observed` for these outcomes. Game outcomes without a specified effect keep the model's own condition differences.

## Scales with reverse-keyed items

The simulator includes participants who ignore item direction, in line with survey research (roughly 10 to 15% of respondents, more among careless responders). On a scale with reverse-keyed items this attenuates the observed effect on the scale mean, as it does in real data. `<Scale>_mean` is computed after reverse-scoring, so it measures the effect as a researcher would analyze it.
