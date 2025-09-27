"""
analysis_notebook.py
====================

This Python script contains all of the analysis code used for the CHASE 2026
submission titled **“Auditing Representation Bias in AI‑Generated Images of
Software Engineers: An Empirical Mixed‑Methods Study.”**  The script is
designed to run in a Jupyter/Google Colab environment and produces the
statistics, tables and charts referenced in the manuscript.  It reads the
provided `All Data.csv` file, cleans and transforms the data, computes
frequency distributions across demographic attributes and software engineering
roles, performs statistical tests (chi‑square and binomial), and generates
figures saved to the local directory.  At the end of the script the key
dataframes are printed so that interactive inspection is possible when the
notebook is executed.

Usage
-----
Upload `All Data.csv` into the Colab session (or place it in the same
directory as this script).  Then run the cells sequentially.  The script
produces charts in `./figures/` and saves CSV summaries in the current
directory.  All dependencies (pandas, numpy, matplotlib, seaborn and
scipy) are part of the Colab Python runtime.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, binomtest

## Load data

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset and perform minimal cleaning.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the coded image annotations.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with stripped whitespace.
    """
    df = pd.read_csv(csv_path)
    # Strip whitespace in string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


## Helper functions

def count_props(prop_str: str) -> int:
    """Count the number of props listed in the Props_Count column.

    The input strings are formatted like "Laptop: 1, Monitors: 2, Whiteboard: 1" or
    "None".  Each comma‑separated entry counts as one prop, regardless of the
    numeric quantity.  If the entry is "None" or missing, returns 0.

    Parameters
    ----------
    prop_str : str
        String representation of the props counts.

    Returns
    -------
    int
        Number of distinct props mentioned in the string.
    """
    if not prop_str or prop_str.lower() in ["none", "nan"]:
        return 0
    items = [item.strip() for item in prop_str.split(',') if item.strip()]
    return len(items)


def binary_tech_presence(tech_str: str) -> int:
    """Convert the Technology_Presence column to a binary indicator.

    Parameters
    ----------
    tech_str : str
        Technology presence description (e.g., "Laptop, Monitor, Code on screens").

    Returns
    -------
    int
        1 if any technology is mentioned, 0 otherwise.
    """
    if not tech_str or tech_str.lower() in ["none", "nan"]:
        return 0
    return 1


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare additional derived columns for analysis.

    This function creates numeric columns for props and binary technology
    presence, and ensures categorical fields are in a consistent format.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with added columns.
    """
    df = df.copy()
    df['Props_Count_Num'] = df['Props_Count'].apply(count_props)
    df['Tech_Presence_Binary'] = df['Technology_Presence'].apply(binary_tech_presence)
    # Normalize age categories
    df['Perceived_Age'] = df['Perceived_Age'].replace({
        'middle-aged': 'Middle-aged',
        'middle aged': 'Middle-aged',
        'older': 'Older',
        'young': 'Young'
    }).fillna('Unknown')
    return df


## Statistical summary functions

def summarize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall counts and percentages for gender, race and age.

    Returns a dataframe with categories and counts/percentages.
    """
    summary = {}
    for attr in ['Perceived_Gender', 'Perceived_Race_Ethnicity', 'Perceived_Age']:
        counts = df[attr].value_counts().reset_index()
        counts.columns = [attr, 'Count']
        counts['Percent'] = counts['Count'] / len(df) * 100
        summary[attr] = counts
    return summary


def summarize_by_role(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Compute counts of a given attribute per role.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the coded images.
    attribute : str
        Column name of the attribute to summarize (e.g., Perceived_Gender).

    Returns
    -------
    pd.DataFrame
        A pivot table with roles as rows and attribute categories as columns.
    """
    pivot = pd.pivot_table(df, index='Role', columns=attribute, aggfunc='size', fill_value=0)
    pivot['Total'] = pivot.sum(axis=1)
    return pivot


def chi_square_test(table: pd.DataFrame) -> tuple:
    """Perform chi‑square test of independence on a contingency table.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table (counts) with categories in rows and columns.

    Returns
    -------
    tuple
        (chi2 statistic, p‑value, degrees of freedom, expected frequencies).
    """
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, expected


def binomial_test_proportion(successes: int, trials: int, p: float) -> float:
    """Compute binomial test p‑value for observing successes in trials given baseline p.

    Parameters
    ----------
    successes : int
        Number of observed successes (e.g., number of female images).
    trials : int
        Total number of observations.
    p : float
        Baseline probability under the null hypothesis.

    Returns
    -------
    float
        Two‑sided p‑value from the binomial test.
    """
    # Use scipy.stats.binomtest for compatibility with newer SciPy versions.
    test = binomtest(successes, trials, p, alternative='two-sided')
    return test.pvalue


## Plotting functions

def plot_gender_by_role(df: pd.DataFrame, output_dir: str = 'figures') -> None:
    """Generate a stacked bar chart showing gender distribution per role.

    Saves the figure as PNG into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = summarize_by_role(df, 'Perceived_Gender')
    # Remove total column for plotting
    pivot_plot = pivot.drop(columns=['Total'])
    # Compute percentages per role
    pivot_pct = pivot_plot.divide(pivot_plot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(12, 6))
    pivot_pct.sort_index().plot(kind='bar', stacked=True)
    plt.title('Gender distribution by software engineering role')
    plt.ylabel('Percentage of images')
    plt.xlabel('Role')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Perceived Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    file_path = os.path.join(output_dir, 'gender_by_role.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_race_by_role(df: pd.DataFrame, output_dir: str = 'figures') -> None:
    """Generate a stacked bar chart showing race/ethnicity distribution per role.

    Saves the figure as PNG into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = summarize_by_role(df, 'Perceived_Race_Ethnicity')
    pivot_plot = pivot.drop(columns=['Total'])
    pivot_pct = pivot_plot.divide(pivot_plot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(12, 6))
    pivot_pct.sort_index().plot(kind='bar', stacked=True, colormap='tab20')
    plt.title('Race/Ethnicity distribution by software engineering role')
    plt.ylabel('Percentage of images')
    plt.xlabel('Role')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Perceived Race/Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    file_path = os.path.join(output_dir, 'race_by_role.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_age_by_role(df: pd.DataFrame, output_dir: str = 'figures') -> None:
    """Generate a stacked bar chart showing age distribution per role.

    Saves the figure as PNG into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    pivot = summarize_by_role(df, 'Perceived_Age')
    pivot_plot = pivot.drop(columns=['Total'])
    pivot_pct = pivot_plot.divide(pivot_plot.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(12, 6))
    pivot_pct.sort_index().plot(kind='bar', stacked=True, colormap='Pastel1')
    plt.title('Age group distribution by software engineering role')
    plt.ylabel('Percentage of images')
    plt.xlabel('Role')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Perceived Age', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    file_path = os.path.join(output_dir, 'age_by_role.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_props_by_gender(df: pd.DataFrame, output_dir: str = 'figures') -> None:
    """Create a boxplot comparing the number of props across genders.

    Saves the figure as PNG into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Perceived_Gender', y='Props_Count_Num', order=['Male','Female','Ambiguous'])
    plt.title('Distribution of props per image across genders')
    plt.xlabel('Perceived Gender')
    plt.ylabel('Number of props')
    plt.tight_layout()
    file_path = os.path.join(output_dir, 'props_by_gender.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_tech_presence_by_model(df: pd.DataFrame, output_dir: str = 'figures') -> None:
    """Plot the proportion of images with visible technology per model.

    Saves the figure as PNG into the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    tech_by_model = df.groupby('LLM_Model')['Tech_Presence_Binary'].agg(['mean','count'])
    # Convert mean to percentage
    tech_by_model['percent_with_tech'] = tech_by_model['mean'] * 100
    plt.figure(figsize=(6, 4))
    sns.barplot(x=tech_by_model.index, y=tech_by_model['percent_with_tech'])
    plt.title('Technology presence by generative model')
    plt.ylabel('Images with visible tech (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.tight_layout()
    file_path = os.path.join(output_dir, 'tech_presence_by_model.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


## Main analysis routine

def main():
    # Specify path to the CSV file
    csv_path = 'All Data.csv'
    df = load_data(csv_path)
    df = prepare_data(df)
    # Overall summaries
    summary = summarize_demographics(df)
    print("Overall demographic breakdown:\n")
    for attr, table in summary.items():
        print(f"\nAttribute: {attr}\n", table)
    # Gender by role
    gender_by_role = summarize_by_role(df, 'Perceived_Gender')
    print("\nGender counts by role:\n", gender_by_role)
    # Race by role
    race_by_role = summarize_by_role(df, 'Perceived_Race_Ethnicity')
    print("\nRace/ethnicity counts by role:\n", race_by_role)
    # Age by role
    age_by_role = summarize_by_role(df, 'Perceived_Age')
    print("\nAge group counts by role:\n", age_by_role)
    # Chi‑square tests for independence
    chi2_gender_role, p_gender_role, dof_gender_role, expected_gender_role = chi_square_test(gender_by_role.drop(columns=['Total']))
    print(f"\nChi‑square test for Gender vs Role: chi2={chi2_gender_role:.2f}, p={p_gender_role:.4g}, dof={dof_gender_role}")
    chi2_race_role, p_race_role, dof_race_role, expected_race_role = chi_square_test(race_by_role.drop(columns=['Total']))
    print(f"Chi‑square test for Race vs Role: chi2={chi2_race_role:.2f}, p={p_race_role:.4g}, dof={dof_race_role}")
    # Binomial tests against real‑world baselines
    # Baseline probabilities from BLS data (2024): 20.3% women, 54.2% White, 6.2% Black, 36.8% Asian, 5.7% Hispanic
    baseline_gender_female = 0.203
    baseline_race_white = 0.542
    baseline_race_black = 0.062
    baseline_race_asian = 0.368
    baseline_race_hispanic = 0.057
    n_total = len(df)
    n_female = (df['Perceived_Gender'] == 'Female').sum()
    pval_gender = binomial_test_proportion(n_female, n_total, baseline_gender_female)
    print(f"\nBinomial test for female representation vs baseline 20.3%: observed {n_female}/{n_total}, p-value={pval_gender:.4g}")
    # Race counts for baseline comparison
    n_white = (df['Perceived_Race_Ethnicity'] == 'White/Caucasian').sum()
    n_black = (df['Perceived_Race_Ethnicity'] == 'Black/African-descent').sum()
    n_asian = ((df['Perceived_Race_Ethnicity'] == 'East Asian') | (df['Perceived_Race_Ethnicity'] == 'South Asian')).sum()
    n_hispanic = (df['Perceived_Race_Ethnicity'] == 'Hispanic/Latinx').sum()
    pval_white = binomial_test_proportion(n_white, n_total, baseline_race_white)
    pval_black = binomial_test_proportion(n_black, n_total, baseline_race_black)
    pval_asian = binomial_test_proportion(n_asian, n_total, baseline_race_asian)
    pval_hispanic = binomial_test_proportion(n_hispanic, n_total, baseline_race_hispanic)
    print(f"\nBinomial tests for race representation vs BLS baselines:")
    print(f"  White: observed {n_white}/{n_total}, baseline 54.2%, p-value={pval_white:.4g}")
    print(f"  Black: observed {n_black}/{n_total}, baseline 6.2%, p-value={pval_black:.4g}")
    print(f"  Asian (East+South): observed {n_asian}/{n_total}, baseline 36.8%, p-value={pval_asian:.4g}")
    print(f"  Hispanic: observed {n_hispanic}/{n_total}, baseline 5.7%, p-value={pval_hispanic:.4g}")
    # Identify roles with zero female images
    female_counts = gender_by_role['Female'] if 'Female' in gender_by_role.columns else pd.Series(0, index=gender_by_role.index)
    zero_female_roles = female_counts[female_counts == 0].index.tolist()
    print("\nRoles with no female depictions:", zero_female_roles)
    # Props by gender summary
    props_by_gender = df.groupby('Perceived_Gender')['Props_Count_Num'].agg(['mean','median','count'])
    print("\nProps count statistics by gender:\n", props_by_gender)
    # Technology presence by gender
    tech_by_gender = df.groupby('Perceived_Gender')['Tech_Presence_Binary'].mean() * 100
    print("\nPercentage of images with visible technology by gender:\n", tech_by_gender)
    # Generate plots
    plot_gender_by_role(df)
    plot_race_by_role(df)
    plot_age_by_role(df)
    plot_props_by_gender(df)
    plot_tech_presence_by_model(df)
    # Save summary tables
    gender_by_role.to_csv('gender_by_role.csv')
    race_by_role.to_csv('race_by_role.csv')
    age_by_role.to_csv('age_by_role.csv')
    props_by_gender.to_csv('props_by_gender.csv')
    tech_by_gender.to_csv('tech_by_gender.csv')
    print("\nAnalysis complete. Summary tables and figures saved.")


if __name__ == '__main__':
    main()