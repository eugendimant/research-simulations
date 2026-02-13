from setuptools import setup, find_packages

setup(
    name="socsim",
    version="0.16.0",
    description="Evidence-traceable social science behavior simulator (games + causal context + atomic evidence + corpus bootstrap).",
    packages=find_packages(),
    install_requires=[
        "openpyxl>=3.1",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "pytest>=7.0",
        "numpy>=1.23",
        "jsonschema>=4.0",
        "requests>=2.31",
    ],
    python_requires=">=3.10",
)
