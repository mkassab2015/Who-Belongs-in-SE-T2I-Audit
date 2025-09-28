#!/usr/bin/env python3
"""
Fairness Metrics Script
-----------------------
Computes Risk Ratios (with 95% CIs), Jensen–Shannon Divergence, and Theil Index
for Gender, Age, and Race distributions of generated images vs. occupational baselines.

Usage:
    python fairness_metrics.py
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.spatial import distance

# -----------------------------
# Baselines (explicit values)
# -----------------------------
baseline_gender = {"Female": 0.203, "Male": 0.797}
baseline_age = {"<40": 0.854, "40-60": 0.139, ">60": 0.007}
baseline_race = {"White": 0.542, "Asian": 0.368, "Black": 0.062, "Other": 0.028}

# -----------------------------
# Helper functions
# -----------------------------
def risk_ratio_ci(n_obs, n_total, p_base, alpha=0.05):
    """Compute Risk Ratio (RR) and Wald 95% CI."""
    p_obs = n_obs / n_total if n_total > 0 else 0
    if p_base == 0:
        return np.nan, (np.nan, np.nan), p_obs
    rr = p_obs / p_base if p_base > 0 else np.nan
    if n_obs == 0 or n_obs == n_total:
        ci = (0.0, np.inf) if n_obs == 0 else (0.0, 0.0)
        return rr, ci, p_obs
    se = np.sqrt(1/n_obs - 1/n_total)
    z = norm.ppf(1 - alpha/2)
    log_rr = np.log(rr)
    ci_low = np.exp(log_rr - z*se)
    ci_high = np.exp(log_rr + z*se)
    return rr, (ci_low, ci_high), p_obs

def jsd(p, q):
    """Jensen-Shannon Divergence (log base 2)."""
    p = np.array(p, dtype=float) / np.sum(p)
    q = np.array(q, dtype=float) / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * distance.rel_entr(p, m).sum()/np.log(2) + \           0.5 * distance.rel_entr(q, m).sum()/np.log(2)

def theil(p, q):
    """Theil index relative to baseline q."""
    p = np.array(p, dtype=float) / np.sum(p)
    q = np.array(q, dtype=float) / np.sum(q)
    return (p * np.log((p + 1e-12) / (q + 1e-12))).sum()

# -----------------------------
# Load Data
# -----------------------------
# Gender data (replace with your gender counts table)
gender_counts = {
    "Qwen-3": {"Female": 0, "Male": 220},
    "Stable Diffusion": {"Female": 49, "Male": 51},
    "GPT-4o": {"Female": 22, "Male": 78},
    "Llama-4": {"Female": 9, "Male": 91},
}

# Age data
age_df = pd.read_csv("Age per LLM.csv")
age_df = age_df.rename(columns={age_df.columns[0]: "Model",
                                age_df.columns[1]: "AgeGroup",
                                age_df.columns[2]: "Count"})
age_df["Model"] = age_df["Model"].ffill()

def map_age_group(x):
    if x == "Young": return "<40"
    elif x == "Middle-Aged": return "40-60"
    elif x == "Older": return ">60"
    else: return None

age_df["Bin"] = age_df["AgeGroup"].map(map_age_group)
age_counts = age_df.groupby(["Model","Bin"])["Count"].sum().unstack().fillna(0).astype(int).to_dict("index")
age_counts["GPT-4o"] = {"<40": 190, "40-60": 16, ">60": 14}

# Race data
race_data = {}
current_model = None
with open("Race per LLM.csv","r",encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split(",") if p.strip()]
        if not parts: continue
        if parts[0] not in ["SA","EA","B","M","L","X","W"]:
            current_model = parts[0]
            race_data[current_model] = {}
            subgroup, count = parts[1], int(parts[2])
            race_data[current_model][subgroup] = count
        else:
            subgroup, count = parts[0], int(parts[1])
            race_data[current_model][subgroup] = count

def aggregate_race_counts(subdict):
    return {
        "White": subdict.get("W",0),
        "Asian": subdict.get("SA",0)+subdict.get("EA",0),
        "Black": subdict.get("B",0),
        "Other": subdict.get("M",0)+subdict.get("L",0)+subdict.get("X",0)
    }

race_counts = {model: aggregate_race_counts(subdict) for model, subdict in race_data.items()}

# -----------------------------
# Compute Metrics
# -----------------------------
def compute_rrs(counts_dict, baselines):
    results = {}
    total = sum(counts_dict.values())
    for subgroup, n in counts_dict.items():
        rr, ci, p_obs = risk_ratio_ci(n, total, baselines[subgroup])
        results[subgroup] = {"RR": rr, "CI": ci, "Obs%": p_obs}
    return results

def compute_divergences(counts_dict, baselines):
    p = np.array([counts_dict[g] for g in baselines.keys()], dtype=float)
    q = np.array([baselines[g] for g in baselines.keys()], dtype=float)
    return jsd(p,q), theil(p,q)

# Compute all results
results = {}
for model in gender_counts.keys():
    results[model] = {
        "Gender": compute_rrs(gender_counts[model], baseline_gender),
        "Age": compute_rrs(age_counts[model], baseline_age),
        "Race": compute_rrs(race_counts[model], baseline_race),
        "DivGender": compute_divergences(gender_counts[model], baseline_gender),
        "DivAge": compute_divergences(age_counts[model], baseline_age),
        "DivRace": compute_divergences(race_counts[model], baseline_race),
    }

# -----------------------------
# Pretty-print summary
# -----------------------------
for model, res in results.items():
    print(f"\n=== {model} ===")
    for dim in ["Gender","Age","Race"]:
        print(f"\n{dim} RRs:")
        for subgroup, vals in res[dim].items():
            rr, ci = vals["RR"], vals["CI"]
            print(f"  {subgroup}: RR={rr:.2f}, CI=[{ci[0]:.2f},{ci[1]:.2f}]")
    for dim, divs in [("Gender",res["DivGender"]),("Age",res["DivAge"]),("Race",res["DivRace"])]:
        print(f"{dim} JSD={divs[0]:.3f}, Theil={divs[1]:.3f}")
