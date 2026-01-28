# Behavioral Experiment Simulation Tool

A Streamlit application for generating synthetic behavioral experiment data using a standardized, persona-driven simulation workflow.

## Features

- **Intuitive Interface**: Step-by-step workflow with progress tracking
- **QSF Support**: Upload and parse Qualtrics Survey Format files
- **Theory-Grounded Simulation**: Uses behavioral personas for realistic data variance
- **Tamper-Proof Audit Log**: PDF documentation of all simulation parameters
- **ZIP Package Output**: Complete deliverable package (CSV, explainer, audit log)
- **Instructor Backend**: Automatic storage of generated files for verification

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

### Streamlit Community Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [Streamlit Community Cloud](https://share.streamlit.io)
3. Click "New app" and select this repository
4. Set the main file path to `simulation_app/app.py`
5. Deploy!

## Usage

### For Students

1. **Team Info**: Enter your group number and team member names
2. **Project Details**: Describe your experiment
3. **Upload Files**: Upload your Qualtrics QSF file and survey screenshots
4. **Define Conditions**: Specify your experimental factors and levels
5. **Configure Variables**: Set up your measurement scales
6. **Set Parameters**: Adjust simulation settings (demographics, attention rates)
7. **Generate**: Create your simulation package

### For Instructors

Generated files are automatically saved to `instructor_copies/` for verification.
Each ZIP includes:
- `Simulated.csv`: The generated dataset
- `Column_Explainer.txt`: Description of all variables
- `Audit_Log.pdf`: Tamper-proof record of all inputs
- `metadata.json`: Machine-readable simulation parameters

## Directory Structure

```
simulation_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/
│   ├── __init__.py       # Package exports
│   ├── qsf_parser.py     # QSF file parsing
│   ├── simulation_engine.py  # Core simulation logic
│   ├── pdf_generator.py  # Audit log PDF generation
│   └── schema_validator.py   # Data validation
└── instructor_copies/    # Backend storage (created at runtime)
```

## Methodology

This tool implements the simulation methodology from "Simulating Behavioral Experiments with ChatGPT-5":

1. **FILE READ OK**: Parse and validate survey structure from QSF
2. **SCHEMA LOCKED**: Define variable schema based on survey elements
3. **PERSONA LIBRARY**: Theory-grounded participant heterogeneity
4. **RESPONSE GENERATION**: Realistic scale responses with appropriate variance
5. **QUALITY ASSURANCE**: Validation checks on generated data

## Credits

- **Instructor**: Prof. Dr. Eugen Dimant
- **Based on**: "Simulating Behavioral Experiments with ChatGPT-5" methodology

## License

For academic use.
