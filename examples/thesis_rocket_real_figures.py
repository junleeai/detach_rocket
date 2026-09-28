from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Files
# ============================================================

FULL_FILE = Path(
    "results/thesis_rocket_real_baseline.csv"
)

FINAL_FILE = Path(
    "results/thesis_rocket_real_subject_selected_final.csv"
)

OUTPUT_DIR = Path(
    "results/figures"
)

SUMMARY_FILE = Path(
    "results/thesis_rocket_real_summary.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load results
# ============================================================

full = pd.read_csv(
    FULL_FILE
)

final = pd.read_csv(
    FINAL_FILE
)


# ============================================================
# Check folds
# ============================================================

expected_folds = np.arange(
    1,
    6,
)

if not np.array_equal(
    full["outer_fold"].to_numpy(),
    expected_folds,
):
    raise RuntimeError(
        "Unexpected Full ROCKET fold ordering."
    )

if not np.array_equal(
    final["outer_fold"].to_numpy(),
    expected_folds,
):
    raise RuntimeError(
        "Unexpected final fold ordering."
    )


# ============================================================
# Extract per-fold values
# ============================================================

folds = expected_folds

full_task = (
    full[
        "subject_balanced_accuracy"
    ]
    .to_numpy()
)

standard_task = (
    final[
        "standard_subject_balanced_accuracy"
    ]
    .to_numpy()
)

identity_task = (
    final[
        "identity_subject_balanced_accuracy"
    ]
    .to_numpy()
)


full_identity = (
    full[
        "identity_accuracy"
    ]
    .to_numpy()
)

standard_identity = (
    final[
        "standard_identity_accuracy"
    ]
    .to_numpy()
)

identity_identity = (
    final[
        "identity_identity_accuracy"
    ]
    .to_numpy()
)

identity_chance = float(
    final[
        "identity_chance"
    ].mean()
)


standard_retained = (
    final[
        "standard_actual_retained_percentage"
    ]
    .to_numpy()
)

identity_retained = (
    final[
        "identity_actual_retained_percentage"
    ]
    .to_numpy()
)


# ============================================================
# Helper
# ============================================================

def mean_sd(
    values,
):
    return (
        float(
            np.mean(values)
        ),
        float(
            np.std(
                values,
                ddof=1,
            )
        ),
    )


# ============================================================
# Summary statistics
# ============================================================

full_task_mean, full_task_sd = mean_sd(
    full_task
)

standard_task_mean, standard_task_sd = mean_sd(
    standard_task
)

identity_task_mean, identity_task_sd = mean_sd(
    identity_task
)


full_identity_mean, full_identity_sd = mean_sd(
    full_identity
)

standard_identity_mean, standard_identity_sd = mean_sd(
    standard_identity
)

identity_identity_mean, identity_identity_sd = mean_sd(
    identity_identity
)


standard_retained_mean, standard_retained_sd = mean_sd(
    standard_retained
)

identity_retained_mean, identity_retained_sd = mean_sd(
    identity_retained
)


# ============================================================
# Summary table
# ============================================================

summary = pd.DataFrame(
    [
        {
            "method": "Full ROCKET",
            "subject_task_mean": full_task_mean,
            "subject_task_sd": full_task_sd,
            "identity_mean": full_identity_mean,
            "identity_sd": full_identity_sd,
            "retained_mean_percent": 100.0,
            "retained_sd_percent": 0.0,
        },
        {
            "method": "Standard SFD",
            "subject_task_mean": standard_task_mean,
            "subject_task_sd": standard_task_sd,
            "identity_mean": standard_identity_mean,
            "identity_sd": standard_identity_sd,
            "retained_mean_percent": standard_retained_mean,
            "retained_sd_percent": standard_retained_sd,
        },
        {
            "method": "Identity-aware SFD",
            "subject_task_mean": identity_task_mean,
            "subject_task_sd": identity_task_sd,
            "identity_mean": identity_identity_mean,
            "identity_sd": identity_identity_sd,
            "retained_mean_percent": identity_retained_mean,
            "retained_sd_percent": identity_retained_sd,
        },
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print(
    "\n============================================================"
)

print(
    "ROCKET REAL-DATA SUMMARY"
)

print(
    "============================================================"
)

display_table = summary.copy()

display_table[
    "subject_task"
] = (
    display_table[
        "subject_task_mean"
    ]
    .mul(100)
    .map(
        lambda x: f"{x:.2f}%"
    )
    +
    " ± "
    +
    display_table[
        "subject_task_sd"
    ]
    .mul(100)
    .map(
        lambda x: f"{x:.2f}%"
    )
)

display_table[
    "identity_leakage"
] = (
    display_table[
        "identity_mean"
    ]
    .mul(100)
    .map(
        lambda x: f"{x:.2f}%"
    )
    +
    " ± "
    +
    display_table[
        "identity_sd"
    ]
    .mul(100)
    .map(
        lambda x: f"{x:.2f}%"
    )
)

display_table[
    "features_retained"
] = (
    display_table[
        "retained_mean_percent"
    ]
    .map(
        lambda x: f"{x:.3f}%"
    )
    +
    " ± "
    +
    display_table[
        "retained_sd_percent"
    ]
    .map(
        lambda x: f"{x:.3f}%"
    )
)

print(
    display_table[
        [
            "method",
            "subject_task",
            "identity_leakage",
            "features_retained",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Figure 1:
# Subject-level task performance by fold
# ============================================================

x = np.arange(
    len(folds)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(
        9,
        5.5,
    )
)

ax.bar(
    x - width,
    full_task * 100,
    width,
    label="Full ROCKET",
)

ax.bar(
    x,
    standard_task * 100,
    width,
    label="Standard SFD",
)

ax.bar(
    x + width,
    identity_task * 100,
    width,
    label="Identity-aware SFD",
)

ax.set_xlabel(
    "Outer subject fold"
)

ax.set_ylabel(
    "Subject-level balanced accuracy (%)"
)

ax.set_title(
    "AD vs Control Classification on Held-Out Subjects"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    folds
)

ax.set_ylim(
    0,
    100,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_subject_task_accuracy.png",
    dpi=300,
)

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_subject_task_accuracy.pdf",
)

plt.close(
    fig
)


# ============================================================
# Figure 2:
# Identity leakage by fold
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        9,
        5.5,
    )
)

ax.bar(
    x - width,
    full_identity * 100,
    width,
    label="Full ROCKET",
)

ax.bar(
    x,
    standard_identity * 100,
    width,
    label="Standard SFD",
)

ax.bar(
    x + width,
    identity_identity * 100,
    width,
    label="Identity-aware SFD",
)

ax.axhline(
    identity_chance * 100,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Chance "
        f"({identity_chance * 100:.2f}%)"
    ),
)

ax.set_xlabel(
    "Outer subject fold"
)

ax.set_ylabel(
    "Subject-identity probe accuracy (%)"
)

ax.set_title(
    "Subject-Identity Leakage on Held-Out Subjects"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    folds
)

ax.set_ylim(
    0,
    100,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_identity_leakage.png",
    dpi=300,
)

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_identity_leakage.pdf",
)

plt.close(
    fig
)


# ============================================================
# Figure 3:
# Feature retention
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)

ax.bar(
    x - width / 2,
    standard_retained,
    width,
    label="Standard SFD",
)

ax.bar(
    x + width / 2,
    identity_retained,
    width,
    label="Identity-aware SFD",
)

ax.set_xlabel(
    "Outer subject fold"
)

ax.set_ylabel(
    "ROCKET features retained (%)"
)

ax.set_title(
    "Feature Retention After SFD"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    folds
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_feature_retention.png",
    dpi=300,
)

fig.savefig(
    OUTPUT_DIR
    / "rocket_real_feature_retention.pdf",
)

plt.close(
    fig
)


# ============================================================
# Done
# ============================================================

print(
    "\nSaved summary table to:"
)

print(
    SUMMARY_FILE
)

print(
    "\nSaved figures:"
)

print(
    OUTPUT_DIR
    / "rocket_real_subject_task_accuracy.png"
)

print(
    OUTPUT_DIR
    / "rocket_real_identity_leakage.png"
)

print(
    OUTPUT_DIR
    / "rocket_real_feature_retention.png"
)

print(
    "\nPDF versions were also saved."
)