# Comparison Report: Karlijn (ChatGPT) vs Claude Simulation

## Executive Summary

This report compares two simulated datasets for the same behavioral experiment (AI Product Recommendations and Psychological Ownership Study):
- **Karlijn's simulation**: Generated using ChatGPT-5, N=400
- **Claude's simulation**: Generated using Claude Opus 4.5, N=300

Both simulations followed the methodology outlined in "Simulating Behavioral Experiments with ChatGPT-5" (BDS5010, Spring 2026).

---

## 1. Structural Differences

| Metric | Karlijn (ChatGPT) | Claude |
|--------|-------------------|--------|
| **Total Rows** | 400 | 300 |
| **Total Columns** | 29 | 24 |
| **Participants per Condition** | 100 | 75 |

**Additional columns in Karlijn's dataset:**
- `PRODUCT_SHOWN` (Body wash / Chocolate bar)
- `SPOT_AI_CORRECT` (binary correctness indicator)
- `AUCTION_CORRECT` (binary correctness indicator)
- `MTURK_ID` (simulated IDs)
- `COMPLETION_CODE`

---

## 2. Demographic Comparison

### Age Distribution
| Statistic | Karlijn | Claude |
|-----------|---------|--------|
| Mean | 37.80 | 34.70 |
| Std Dev | 11.44 | 11.55 |
| Min | 18 | 18 |
| Max | 65 | 65 |

**Observation:** Both simulations produce realistic age distributions for US adults 18-65. Karlijn's mean is slightly higher.

### Gender Distribution
| Gender | Karlijn | Claude |
|--------|---------|--------|
| Male | 196 (49.0%) | 170 (56.7%) |
| Female | 196 (49.0%) | 121 (40.3%) |
| Non-binary | 4 (1.0%) | 3 (1.0%) |
| Prefer not to say | 4 (1.0%) | 6 (2.0%) |

**Observation:** Karlijn achieves perfect 50/50 gender balance. Claude shows slight male overrepresentation but maintains the ~2% minority gender category as specified.

---

## 3. Attention/Manipulation Checks

| Check | Karlijn | Claude |
|-------|---------|--------|
| **AI Mention Check Accuracy** | 49.2% | 90.0% |
| **Auction Winner Check Accuracy** | 87.2% | 86.3% |

### Critical Finding: AI Mention Check

**Karlijn's 49.2% accuracy is problematic.** This suggests participants are responding at near-chance levels to whether AI was mentioned on the shopping page. In a real experiment:
- AI conditions should answer "Yes" (~95%)
- No AI conditions should answer "No" (~95%)
- Expected overall accuracy: ~95% (with ~5% inattentive responders)

**Claude's 90.0% accuracy** is more realistic, reflecting a small proportion of inattentive participants who fail the manipulation check.

**The auction winner comprehension check** shows comparable performance (87.2% vs 86.3%), indicating similar attention to instructions.

---

## 4. Dependent Variable: Willingness to Pay (WTP)

### Overall Statistics
| Statistic | Karlijn | Claude |
|-----------|---------|--------|
| Mean | 5.62 | 5.50 |
| Std Dev | 2.16 | 1.53 |
| Median | 6.0 | 5.5 |

### WTP by Condition
| Condition | Karlijn M (SD) | Claude M (SD) |
|-----------|----------------|---------------|
| No AI x Utilitarian | 5.46 (2.11) | 5.11 (1.39) |
| No AI x Hedonic | 6.01 (2.19) | 5.80 (1.58) |
| AI x Utilitarian | 5.43 (2.11) | 5.69 (1.58) |
| AI x Hedonic | 5.56 (2.20) | 5.39 (1.50) |

**Observations:**
1. Both simulations produce similar overall WTP means (~5.5 on 0-10 scale)
2. Karlijn shows higher variance (SD ~2.1) vs Claude (SD ~1.5)
3. Both show slightly higher WTP for hedonic (Snickers) vs utilitarian (Dove) products in the No AI condition
4. Neither simulation shows strong AI effects on WTP (hypothesis-neutral)

---

## 5. Psychological Ownership Scale

| Item | Karlijn M (SD) | Claude M (SD) |
|------|----------------|---------------|
| Incorporates self | 2.00 (0.00) | 3.42 (1.17) |
| Belongs to me | 1.96 (0.19) | 3.57 (1.26) |
| Connected | 1.53 (0.50) | 3.45 (1.24) |
| Closeness | 1.05 (0.21) | 3.35 (1.24) |
| Difficult (reverse) | 3.46 (0.70) | 3.55 (1.36) |
| **Overall Composite** | **2.02 (0.18)** | **3.45 (0.98)** |

### Critical Finding: Variance in Psychological Ownership

**Karlijn's PO responses show extremely low variance:**
- Item 1 has SD=0.00 (all responses identical!)
- Items 2-4 have SD < 0.50
- This is unrealistic for human data

**Claude's PO responses show realistic variance:**
- All items have SD > 1.0
- Composite SD = 0.98 (typical for Likert scales)

**The near-zero variance in Karlijn's data would fail any data quality check in a real study.**

---

## 6. AI Attitudes Scale

| Item | Karlijn M (SD) | Claude M (SD) |
|------|----------------|---------------|
| Improve my life | 1.98 (0.13) | 3.69 (1.33) |
| Improve my work | 1.82 (0.38) | 3.75 (1.37) |
| Use in future | 1.55 (0.50) | 3.67 (1.38) |
| Positive for humanity | 1.19 (0.39) | 3.73 (1.34) |
| Threat to humans | 1.06 (0.24) | 3.27 (1.37) |
| **Overall Composite** | **2.50 (0.16)** | **3.72 (1.19)** |

### Critical Finding: AI Attitudes Compression

**Karlijn's AI attitude responses are heavily compressed:**
- All means are below 2.0 (on a 1-6 scale)
- All SDs are below 0.5
- This suggests almost uniformly negative AI attitudes with no individual differences

**Claude's responses show:**
- Means around the midpoint (3.3-3.8)
- Realistic variance (SD ~1.3)
- Individual differences reflecting the persona library

---

## 7. Hedonic/Utilitarian Manipulation Check

| Product Type | Karlijn M (SD) | Claude M (SD) |
|--------------|----------------|---------------|
| Utilitarian (Dove) | 2.85 (0.76) | 2.52 (1.02) |
| Hedonic (Snickers) | 5.05 (0.79) | 5.45 (1.08) |

**Observation:** Both simulations successfully differentiate hedonic from utilitarian products:
- Utilitarian products rated toward the utilitarian end (1-3)
- Hedonic products rated toward the hedonic end (5-7)
- Claude shows slightly larger separation (2.93 points) vs Karlijn (2.20 points)

---

## 8. Visual Comparison Summary

See accompanying figures:
1. `1_WTP_by_Condition.png` - WTP bar charts by condition
2. `2_WTP_Comparison_SideBySide.png` - Direct WTP comparison
3. `3_WTP_Distribution.png` - WTP histograms
4. `4_PO_by_Condition.png` - Psychological ownership by condition
5. `5_Manipulation_Check.png` - Hedonic/utilitarian ratings
6. `6_AI_Attitudes.png` - AI attitude item comparison
7. `7_Summary_Radar.png` - Overall profile comparison

---

## 9. Key Takeaways

### Similarities
1. **Overall WTP** is comparable (~5.5 on 0-10 scale)
2. **Hedonic/Utilitarian manipulation** works in both simulations
3. **Comprehension check accuracy** is similar (~87%)
4. **Age and gender distributions** are reasonable

### Differences

| Aspect | Karlijn (ChatGPT) | Claude | Better? |
|--------|-------------------|--------|---------|
| **AI Check Accuracy** | 49.2% (near chance) | 90.0% (realistic) | Claude |
| **PO Variance** | Near-zero (unrealistic) | SD ~1.0 (realistic) | Claude |
| **AI Attitude Variance** | Very low (unrealistic) | SD ~1.3 (realistic) | Claude |
| **Response Range Usage** | Compressed to low end | Full range used | Claude |
| **Sample Size** | 400 | 300 | Karlijn |

### Recommendations

1. **For realistic simulation data:** Claude's approach produces more human-like variance patterns
2. **For attention checks:** Karlijn's AI mention check failure is a red flag that should be addressed
3. **For scale responses:** Simulations should use the full response range with realistic variance
4. **For hypothesis testing:** Neither simulation "forces" effects - both maintain hypothesis neutrality

---

## 10. Conclusion

Both ChatGPT-5 and Claude can produce simulated behavioral experiment data, but with notable differences in data quality:

**Claude's simulation** better approximates real human data with:
- Realistic variance in scale responses
- Appropriate attention check accuracy
- Use of full response range
- Theory-grounded persona-based heterogeneity

**Karlijn's ChatGPT simulation** shows concerning patterns:
- Near-zero variance on key scales (unrealistic)
- 49% AI manipulation check accuracy (suggests systematic error)
- Compressed response distributions

For pilot testing and power analysis, **Claude's simulation would provide more reliable estimates** of expected effect sizes and data patterns.

---

*Report generated: 2026-01-27*
*Comparison performed using Python 3.x with pandas and matplotlib*
