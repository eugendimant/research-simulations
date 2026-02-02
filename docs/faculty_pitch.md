# Behavioral Experiment Simulation Tool
## A Teaching Tool for Research Methods Courses

---

**Subject: New Tool for Teaching Experimental Design & Data Analysis**

Hi colleagues,

I've developed a tool that generates realistic experimental data from students' own study designs—bridging the gap between when students design their experiments and when they can actually practice analyzing data.

### The Problem

Students design experiments but can't practice analysis until data collection completes weeks later. They lose momentum and miss the connection between design choices and analytical consequences.

### The Solution

A web-based tool where students upload their Qualtrics survey (.qsf file) and preregistration. The tool produces simulated datasets mirroring what they'd collect from real participants—complete with realistic response patterns, attention check failures, and demographic variation.

### Key Features

- **Automatic Survey Parsing**: Extracts conditions, scales, and structure directly from Qualtrics files
- **Realistic Simulations**: Configurable effect sizes, response styles (satisficers, extreme responders), and exclusion criteria
- **Flexible Designs**: Supports simple comparisons through 3×3×3 factorial designs
- **Instant Instructor Reports**: Comprehensive analysis document generated automatically (see below)

### Automatic Instructor Report

The tool generates a confidential instructor report that serves as your answer key. It automatically computes descriptive statistics by condition, runs the appropriate inferential tests (t-tests, ANOVA, non-parametric alternatives) based on the experimental design, calculates effect sizes (Cohen's d, eta-squared), and checks statistical assumptions (normality, homogeneity of variance). For factorial designs, it performs two-way ANOVA with interaction effects. Each analysis includes publication-ready visualizations—bar charts with confidence intervals, distribution plots, and histograms—along with plain-language interpretations of what the results mean. The report concludes with an executive summary highlighting which hypotheses were supported, the strongest effects observed, and key takeaways, allowing you to quickly assess whether findings align with the study's pre-registered predictions.

### What Students Learn

1. **Design-Analysis Connection**: See how experimental design translates to data structure
2. **Data Cleaning**: Practice identifying exclusions before real stakes
3. **Statistical Analysis**: Run planned analyses on realistic data
4. **Preregistration Value**: Compare analysis plans against what data requires
5. **Troubleshooting**: Catch design flaws before data collection

### Ideal For

- Research methods courses (undergrad or graduate)
- Thesis/dissertation preparation
- Lab meetings for study piloting
- Statistics courses needing realistic datasets

### Next Steps

I'd welcome a 15-minute demo or the chance to have you test it with an existing study design. The tool is browser-based and requires no installation.

Best,
[Your name]

---

*Browser-based (Streamlit) • No installation required • Exports to CSV/Excel/HTML*
