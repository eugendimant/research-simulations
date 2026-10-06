# How effects work

A simulated dataset needs a rule for how conditions differ. This page explains the two kinds of rules the tool uses, how closely the data follow them, and how to see what was built in.

## Effects you specify

Open **Advanced Settings** and add an expected effect size. You choose:

- the dependent variable (a scale from your survey),
- the factor and the two conditions to contrast (which one should score higher),
- the Cohen's d and the direction.

The generator shifts the response tendency of participants in the two conditions, half the gap up and half down. That shift is **calibrated to the scale you are measuring**. A given shift produces a different standardized difference depending on how many items are averaged: items share person-level variance, so the mean of several items has less noise than one item. Without a correction, the d on a ten-item scale mean would be about twice the d on a single item. The calibration makes the requested d the target for the **scale mean** (or for the item, on a single-item scale).

What to expect:

| Question | Answer |
|---|---|
| Which variable has the requested d? | The scale mean (`<Scale>_mean`, which is in `Simulation_Diagnostics.csv`). Individual items show a smaller d, as in real data, because each item carries its own noise. |
| How close does a run get? | Close to the requested value on average, with ordinary sampling variation. One standard error of d is about 0.14 at 100 participants per group and 0.10 at 200 per group. |
| Does it depend on the scale? | The calibration accounts for the number of items, the number of response options (including two- and three-point scales) and very wide scales such as 0-100 sliders. In the cases checked, the average observed d was within 8% of the request (table below). |
| Can other settings move it? | A little. Variables that most people rate near the top of the scale, and willingness-to-pay style variables with a wide spread, pull it toward zero. Random responders and attention-check failures dilute it a little more. |

### How close the calibration gets

Each cell is the observed d divided by the requested d (1.00 is exact), averaged over 6 to 8 independent runs of 1,200 participants each with a requested d of 0.5. These runs used random seeds that were not used to fit the calibration. One run on its own varies around the average (a standard deviation of roughly 0.1 in these units at this sample size) because of ordinary sampling error.

| Scale | 1 item | 3 items | 4-6 items | 8 items | 12 items | 20 items |
|---|---|---|---|---|---|---|
| 2-point (0/1) | 0.96 | 0.93 | 0.93 | | | |
| 3-point | 0.99 | 0.99 | 0.93 | | | |
| 5-point | | | 1.02 | 1.01 | | |
| 7-point | 1.08 | 1.05 | 1.04 | 1.04 | 1.02 | 1.06 |
| 11-point (0-10) | | | 1.08 | 1.04 | | |
| 0-100 slider | 0.98 | | | 1.06 | | |

The 4-6 item column holds 4 items for the 5-point and 11-point rows, 5 for the 7-point row and 6 for the 2-point and 3-point rows.

Before this calibration of long scales, the same 7-point runs gave 1.16 at 8 items, 1.19 at 12, 1.24 at 15 and 1.25 at 20, and 0.71 to 0.75 on a two-point scale. Scales with more than 30 items use the 30-item correction. Effects on scales with reverse-keyed items, with several correlated outcomes, or with many random responders are attenuated further (see below); the table is for a single, plainly keyed scale.


## Effects the tool infers

If you do not specify an effect for a variable, the tool reads the condition names and the study description. A "gain" frame versus a "loss" frame, "AI" versus "human", "high" versus "low", a named paradigm with a published estimate, and similar patterns each get a literature-informed difference, so that conditions do not come out identical.

These inferred effects are **a heuristic, not an effect you chose**:

- They are not tuned to a target d. The observed d can differ from what the wording suggests.
- The total inferred shift is capped, so that keyword combinations cannot produce implausible effects.
- They apply only to variables you did not specify an effect for. Where you specify an effect for a variable, nothing inferred from the condition names is added on top of it as a mean shift. For conditions named in any effect you specify, the name-based changes to response style are switched off as well.

To remove them, untick **Also infer small differences from the condition names** in Advanced Settings. Then only the effects you specify are built in, and every other contrast is a true null apart from sampling noise. This is the setting to use when you want to check a false-positive rate, or when you do not want the tool to guess.

## Seeing what was built in

Every run writes the following into `Metadata.json`:

- `effect_sizes_configured`: the effects you specified.
- `effect_sizes_applied`: for each variable and pair of conditions, whether the contrast came from your specification (`user`), from the name-based heuristic (`inferred`), or from nothing (`none`), the intended d where one exists, and the d observed in this sample.
- `effect_sizes_observed`: the observed Cohen's d for every item and scale mean, for every pair of conditions.

`User_Study_Summary.md` shows the same information as tables.

## Study context versus condition names

Study-level context (for example "this is a political study") nudges every condition's response style in the same way and does not create differences between conditions. Only condition names create differences, and only where you have not specified an effect.

## Economic-game outcomes

Outcomes that look like dictator, trust, ultimatum or public-goods games are generated by a behavioral-economics model, which replaces the numeric generator's values for those columns. The model gives the distribution its realistic shape (for example, clusters at zero and at an even split). If you specify an effect for such an outcome, it is restored after the model runs: each condition's mean is moved to its target (plus or minus half the requested contrast, in standard deviations of the outcome), with values rounded and clipped to the scale range. The observed d is therefore slightly below the request when many answers sit at the scale limits. Check `effect_sizes_observed` for these outcomes. Game outcomes without a specified effect keep the model's own condition differences.

## Scales with reverse-keyed items

The simulator includes participants who ignore item direction, in line with survey research (roughly 10 to 15% of respondents, more among careless responders). On a scale with reverse-keyed items this attenuates the observed effect on the scale mean, as it does in real data. `<Scale>_mean` is computed after reverse-scoring, so it measures the effect as a researcher would analyze it.
