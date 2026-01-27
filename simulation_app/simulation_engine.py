# utils/simulation_engine.py (PATCH)
# Replace your existing generate_r_export method with this version.
# This avoids backslash line-continuations (the Streamlit error you showed is consistent with a "\" that has trailing spaces).

from __future__ import annotations

import pandas as pd


def generate_r_export(self, df: pd.DataFrame) -> str:
    """
    Return an R script that reads Simulated.csv and sets CONDITION as a factor.

    This version uses parentheses for string assembly (no "\" line continuation),
    which prevents: SyntaxError: unexpected character after line continuation character
    """
    # Try to infer the condition column name your engine uses
    condition_col = "CONDITION" if "CONDITION" in df.columns else ("Condition" if "Condition" in df.columns else None)

    levels_line = ""
    if condition_col:
        levels = [str(x) for x in df[condition_col].dropna().unique().tolist()]
        # Stable order
        levels = sorted(levels)
        r_levels = ", ".join([f"'{lvl.replace(\"'\", \"\\\\'\")}'" for lvl in levels])
        levels_line = f"data${condition_col} <- factor(data${condition_col}, levels=c({r_levels}))\n"

    return (
        "suppressPackageStartupMessages({\n"
        "  library(readr)\n"
        "  library(dplyr)\n"
        "})\n\n"
        "data <- read_csv('Simulated.csv', show_col_types = FALSE)\n"
        "data <- data %>% mutate(across(where(is.character), as.factor))\n"
        + levels_line
        + "\n"
        "print(glimpse(data))\n"
    )
