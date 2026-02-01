# Behavioral Experiment Simulation Tool
## A Teaching Tool for Research Methods Courses

---

**Subject: New Tool for Teaching Experimental Design & Data Analysis - Would Love Your Feedback**

Hi colleagues,

I've been developing a tool that I think could be valuable for anyone teaching research methods, experimental design, or data analysis. I'd love to get your thoughts and explore whether it might be useful in your courses.

### What Is It?

A web-based simulation tool that generates realistic experimental data from students' own study designs. Students upload their Qualtrics survey (.qsf file) and preregistration, and the tool produces simulated datasets that mirror what they'd collect from real participants.

### The Problem It Solves

We often face a timing gap: students design experiments but can't practice analysis until data collection completes weeks later. Meanwhile, they lose momentum and the connection between design choices and analytical consequences. This tool bridges that gap.

### Key Capabilities

- **Realistic Data Generation**: Simulates responses with configurable effect sizes, response styles (satisficers, extreme responders, careless respondents), and attention check failures
- **Automatic Survey Parsing**: Extracts conditions, scales, and experimental structure directly from Qualtrics files
- **Flexible Designs**: Supports 2x2 factorials, multi-arm trials, factorial + control designs
- **Built-in Exclusion Criteria**: Generates completion times, straight-lining patterns, and flags for data cleaning practice
- **Comprehensive Instructor Reports**: Automatically runs t-tests, ANOVA, factorial ANOVA with interactions, Mann-Whitney, chi-squared, regression analyses with effect sizes and assumption checks - giving you an answer key

### What Students Learn

1. **Design-Analysis Connection**: See immediately how their experimental design translates to analyzable data
2. **Data Cleaning**: Practice identifying and justifying exclusions before real stakes
3. **Statistical Analysis**: Run their planned analyses on realistic data structures
4. **Preregistration Value**: Compare their analysis plan against what the data actually requires
5. **Troubleshooting**: Catch design flaws (unbalanced conditions, scale issues) before data collection

### For Instructors

The confidential instructor report provides:
- Complete statistical analysis with visualizations
- Effect sizes (Cohen's d, eta-squared) with interpretations
- Assumption checks (Levene's, normality tests)
- Factorial ANOVA with interaction effects
- Pairwise comparisons with forest plots
- An "answer key" to evaluate student work against

### Ideal Use Cases

- Research methods courses (undergrad or PhD)
- Thesis/dissertation preparation
- Lab meetings for study piloting
- Statistics courses needing realistic datasets

### Try It / Next Steps

I'd welcome the chance to:
1. Give you a 15-minute demo
2. Have you test it with an existing study design
3. Discuss how it might fit your course structure

The tool is browser-based (Streamlit) and requires no installation for students. Happy to share access and would genuinely appreciate your feedback on what would make this more useful.

Best,
[Your name]

---

*Built with: Python, Streamlit, NumPy/Pandas, Matplotlib*
*Supports: Qualtrics .qsf files, CSV/Excel export, HTML reports*
