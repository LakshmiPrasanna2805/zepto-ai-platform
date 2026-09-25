"""Shared, deterministic cleaning rules for the Titanic dataset.

Both notebooks import this file so the data is cleaned ONE way, in ONE place:
    01_eda.ipynb       -> basic_clean(df)  +  impute_age_for_eda(df)
    02_modeling.ipynb  -> basic_clean(df)  (age is then imputed INSIDE the
                          scikit-learn Pipeline, fitted on the training split only)

`basic_clean` only uses fixed rules (drop rows / add a category). It never
learns a number (mean, median, ...) from the data, so running it before the
train/test split cannot leak test-set information into training.
"""
import pandas as pd

# Threshold rule from the assignment
DROP_ROWS_BELOW = 5.0     # < 5 % missing        -> drop those rows
IMPUTE_UP_TO = 30.0       # 5 % – 30 % missing   -> impute
                          # > 30 % missing       -> drop column OR encode "missing" as its own category


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values for every column that has any, plus the rule that applies."""
    pct = df.isna().mean().mul(100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)

    def rule(p):
        if p < DROP_ROWS_BELOW:
            return "< 5 %  -> drop rows"
        if p <= IMPUTE_UP_TO:
            return "5-30 % -> impute"
        return "> 30 % -> drop column or 'missing' category"

    return pd.DataFrame({"missing_count": df[pct.index].isna().sum(),
                         "missing_pct": pct,
                         "threshold_rule": [rule(p) for p in pct]})


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based cleaning that learns nothing from the data.

    * embarked / embark_town (0.22 % missing, < 5 %)  -> drop those 2 rows
    * deck (77.22 % missing, > 30 %)                   -> keep, with "Unknown" as its own category
    * age (19.87 % missing, 5-30 %)                    -> left for imputation (see below)
    """
    out = df.copy()
    out = out.dropna(subset=["embarked", "embark_town"]).reset_index(drop=True)
    out["deck"] = out["deck"].astype("object").fillna("Unknown")
    return out


def impute_age_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """EDA-only age imputation: median age of passengers with the same sex and class.

    (Used for exploring the full dataset. The modeling notebook does NOT use this;
    its Pipeline learns the imputation value from the training split only.)
    """
    out = df.copy()
    group_median = out.groupby(["sex", "pclass"])["age"].transform("median")
    out["age"] = out["age"].fillna(group_median)
    return out
