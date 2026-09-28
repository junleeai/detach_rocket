import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

RESULTS_DIR = "results"

FIGURE_DIR = os.path.join(
    RESULTS_DIR,
    "figures",
)

SEED_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_synthetic_seed_summary.csv",
)

VALIDATION_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_synthetic_validation.csv",
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# =========================================================
# Load results
# =========================================================

seed_df = pd.read_csv(
    SEED_SUMMARY_CSV
)

validation_df = pd.read_csv(
    VALIDATION_CSV
)


CONDITION_ORDER = [
    "weak",
    "medium",
    "strong",
]

CONDITION_LABELS = [
    "Low",
    "Medium",
    "High",
]


# =========================================================
# Helpers
# =========================================================

def mean_and_sd(
    dataframe,
    condition,
    column,
):
    values = dataframe.loc[
        dataframe[
            "identity_condition"
        ] == condition,
        column,
    ].to_numpy()

    return (
        np.mean(values),
        np.std(
            values,
            ddof=1,
        ),
    )


def save_figure(
    figure,
    filename,
):
    png_path = os.path.join(
        FIGURE_DIR,
        f"{filename}.png",
    )

    pdf_path = os.path.join(
        FIGURE_DIR,
        f"{filename}.pdf",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


# =========================================================
# Shared x positions
# =========================================================

x = np.arange(
    len(CONDITION_ORDER)
)


# =========================================================
# Figure 1
# Task balanced accuracy
# =========================================================

task_methods = {
    "Full ROCKET":
        "full_task_mean",
    "Standard SFD":
        "standard_task_mean",
    "Identity-aware SFD":
        "identity_task_mean",
}


figure, axis = plt.subplots(
    figsize=(8, 5)
)


for method_name, column in (
    task_methods.items()
):

    means = []
    sds = []

    for condition in CONDITION_ORDER:

        mean_value, sd_value = (
            mean_and_sd(
                seed_df,
                condition,
                column,
            )
        )

        means.append(
            100 * mean_value
        )

        sds.append(
            100 * sd_value
        )

    axis.errorbar(
        x,
        means,
        yerr=sds,
        marker="o",
        capsize=4,
        label=method_name,
    )


axis.set_xticks(
    x
)

axis.set_xticklabels(
    CONDITION_LABELS
)

axis.set_xlabel(
    "Subject-identity signal amplitude"
)

axis.set_ylabel(
    "Task balanced accuracy (%)"
)

axis.set_title(
    "Task performance"
)

# Narrower range makes differences visible while
# still including all means and error bars.
axis.set_ylim(
    55,
    105,
)

axis.legend()

axis.grid(
    axis="y",
    alpha=0.3,
)

figure.tight_layout()


save_figure(
    figure,
    "rocket_synthetic_task_accuracy",
)

plt.close(
    figure
)


# =========================================================
# Figure 2
# Identity leakage
# =========================================================

identity_methods = {
    "Full ROCKET":
        "full_identity_mean",
    "Standard SFD":
        "standard_identity_mean",
    "Identity-aware SFD":
        "identity_identity_mean",
}


figure, axis = plt.subplots(
    figsize=(8, 5)
)


for method_name, column in (
    identity_methods.items()
):

    means = []
    sds = []

    for condition in CONDITION_ORDER:

        mean_value, sd_value = (
            mean_and_sd(
                seed_df,
                condition,
                column,
            )
        )

        means.append(
            100 * mean_value
        )

        sds.append(
            100 * sd_value
        )

    axis.errorbar(
        x,
        means,
        yerr=sds,
        marker="o",
        capsize=4,
        label=method_name,
    )


axis.axhline(
    25,
    linestyle="--",
    label="Identity chance level",
)

axis.set_xticks(
    x
)

axis.set_xticklabels(
    CONDITION_LABELS
)

axis.set_xlabel(
    "Subject-identity signal amplitude"
)

axis.set_ylabel(
    "Identity-probe accuracy (%)"
)

axis.set_title(
    "Subject-identity leakage"
)

axis.set_ylim(
    0,
    105,
)

axis.legend()

axis.grid(
    axis="y",
    alpha=0.3,
)

figure.tight_layout()


save_figure(
    figure,
    "rocket_synthetic_identity_leakage",
)

plt.close(
    figure
)


# =========================================================
# Figure 3
# Feature retention
#
# Individual synthetic-seed means are shown because the
# retained-feature percentages are strongly skewed.
# A logarithmic y-axis avoids misleading negative
# mean-minus-SD error bars.
# =========================================================

retention_methods = {
    "Standard SFD":
        "standard_retained_mean",
    "Identity-aware SFD":
        "identity_retained_mean",
}


figure, axis = plt.subplots(
    figsize=(8, 5)
)


method_offsets = {
    "Standard SFD": -0.08,
    "Identity-aware SFD": 0.08,
}


for method_name, column in (
    retention_methods.items()
):

    offset = method_offsets[
        method_name
    ]

    mean_values = []

    # First compute condition means.
    for condition in CONDITION_ORDER:

        values = seed_df.loc[
            seed_df[
                "identity_condition"
            ] == condition,
            column,
        ].to_numpy()

        mean_values.append(
            np.mean(values)
        )

    # Plot mean line first and obtain its default color.
    mean_line = axis.plot(
        x + offset,
        mean_values,
        marker="D",
        linewidth=2,
        markersize=7,
        label=f"{method_name} mean",
    )[0]

    method_color = (
        mean_line.get_color()
    )

    # Plot the five individual seed values.
    for condition_index, condition in (
        enumerate(CONDITION_ORDER)
    ):

        values = seed_df.loc[
            seed_df[
                "identity_condition"
            ] == condition,
            column,
        ].to_numpy()

        axis.scatter(
            np.full(
                len(values),
                x[condition_index]
                + offset,
            ),
            values,
            alpha=0.55,
            s=40,
            color=method_color,
        )


axis.set_yscale(
    "log"
)

axis.set_xticks(
    x
)

axis.set_xticklabels(
    CONDITION_LABELS
)

axis.set_xlabel(
    "Subject-identity signal amplitude"
)

axis.set_ylabel(
    "ROCKET features retained (%)"
)

axis.set_title(
    "Feature retention"
)

axis.legend()

axis.grid(
    axis="y",
    alpha=0.3,
    which="both",
)

figure.tight_layout()


save_figure(
    figure,
    "rocket_synthetic_feature_retention",
)

plt.close(
    figure
)


# =========================================================
# Figure 4
# Lambda-selection frequencies
# =========================================================

selected_validation = validation_df[
    validation_df[
        "selected"
    ]
    .astype(str)
    .str.lower()
    == "true"
].copy()


lambda_values = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
]


counts = np.zeros(
    (
        len(CONDITION_ORDER),
        len(lambda_values),
    ),
    dtype=int,
)


for condition_index, condition in (
    enumerate(CONDITION_ORDER)
):

    condition_data = (
        selected_validation[
            selected_validation[
                "identity_condition"
            ] == condition
        ]
    )

    for lambda_index, lambda_value in (
        enumerate(lambda_values)
    ):

        counts[
            condition_index,
            lambda_index,
        ] = np.sum(
            np.isclose(
                condition_data[
                    "lambda"
                ].to_numpy(),
                lambda_value,
            )
        )


figure, axis = plt.subplots(
    figsize=(9, 5)
)


bar_width = 0.15

offsets = (
    np.arange(
        len(lambda_values)
    )
    - (
        len(lambda_values) - 1
    ) / 2
) * bar_width


for lambda_index, lambda_value in (
    enumerate(lambda_values)
):

    axis.bar(
        x + offsets[
            lambda_index
        ],
        counts[
            :,
            lambda_index
        ],
        width=bar_width,
        label=(
            f"λ = {lambda_value:g}"
        ),
    )


axis.set_xticks(
    x
)

axis.set_xticklabels(
    CONDITION_LABELS
)

axis.set_xlabel(
    "Subject-identity signal amplitude"
)

axis.set_ylabel(
    "Number of outer folds"
)

axis.set_title(
    "Validation-selected λ"
)

axis.set_ylim(
    0,
    25,
)

axis.legend(
    ncol=3,
)

axis.grid(
    axis="y",
    alpha=0.3,
)

figure.tight_layout()


save_figure(
    figure,
    "rocket_synthetic_lambda_selection",
)

plt.close(
    figure
)


print(
    "\nAll figures generated successfully."
)