"""
Final outer-test evaluation using subject-level model selection.

The lambda and pruning percentages were selected using ONLY the
training and validation subjects in:

    results/thesis_rocket_real_subject_level_selection.csv

For each outer fold:

1. Use all 52 non-test subjects as the development set.
2. Refit Standard SFD using its subject-selected retention percentage.
3. Refit Identity-aware SFD using its subject-selected lambda and
   retention percentage.
4. Evaluate once on the 13 held-out test subjects.

Primary task metric:
    subject-level balanced accuracy

Secondary task metric:
    window-level balanced accuracy

Identity metric:
    subject-identity probe accuracy on the frozen test representation.
"""

from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd

from aeon.transformations.collection.convolution_based import Rocket

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

from detach_rocket.detach_classes import DetachRocket


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(
    "/Users/Jun/Desktop/AI Health/Master Thesis/thesis data/"
    "processed/ds004504_ad_control_10s_125hz.npz"
)

FOLD_FILE = Path(
    "results/thesis_real_subject_folds.csv"
)

SELECTION_FILE = Path(
    "results/thesis_rocket_real_subject_level_selection.csv"
)

FULL_BASELINE_FILE = Path(
    "results/thesis_rocket_real_baseline.csv"
)

RESULT_FILE = Path(
    "results/thesis_rocket_real_subject_selected_final.csv"
)

PREDICTION_FILE = Path(
    "results/"
    "thesis_rocket_real_subject_selected_subject_predictions.csv"
)

N_KERNELS = 10_000
N_FOLDS = 5
RANDOM_SEED = 42


# ============================================================
# Subject-level task evaluation
# ============================================================

def evaluate_subject_level(
    decision_scores,
    y_true,
    subject_ids,
):
    rows = []

    for subject_id in np.unique(
        subject_ids
    ):

        mask = (
            subject_ids == subject_id
        )

        labels = np.unique(
            y_true[mask]
        )

        if len(labels) != 1:
            raise RuntimeError(
                f"{subject_id} has multiple task labels."
            )

        mean_score = float(
            decision_scores[
                mask
            ].mean()
        )

        true_label = int(
            labels[0]
        )

        predicted_label = int(
            mean_score > 0
        )

        rows.append(
            {
                "subject": subject_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "mean_decision_score": mean_score,
            }
        )

    subject_df = pd.DataFrame(
        rows
    )

    score = balanced_accuracy_score(
        subject_df[
            "true_label"
        ],
        subject_df[
            "predicted_label"
        ],
    )

    return (
        float(score),
        subject_df,
    )


# ============================================================
# Identity probe
# ============================================================

def identity_probe_accuracy(
    features,
    subject_labels,
    random_state=42,
):
    unique_subjects = np.unique(
        subject_labels
    )

    chance = (
        1.0
        / len(unique_subjects)
    )

    counts = np.unique(
        subject_labels,
        return_counts=True,
    )[1]

    if not np.all(
        counts >= 5
    ):
        raise RuntimeError(
            "Not enough windows per subject "
            "for 5-fold identity probe."
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    true_labels = []
    predicted_labels = []

    for (
        train_idx,
        test_idx,
    ) in cv.split(
        features,
        subject_labels,
    ):

        probe = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )

        probe.fit(
            features[
                train_idx
            ],
            subject_labels[
                train_idx
            ],
        )

        predictions = probe.predict(
            features[
                test_idx
            ]
        )

        true_labels.extend(
            subject_labels[
                test_idx
            ]
        )

        predicted_labels.extend(
            predictions
        )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    return (
        float(accuracy),
        float(chance),
    )


# ============================================================
# Load data
# ============================================================

data = np.load(
    DATA_FILE
)

X = data["X"]
y_task = data["y_task"]
y_subject = data["y_subject"]
subject_ids = data["subject_ids"]

fold_table = pd.read_csv(
    FOLD_FILE
)

selection_table = pd.read_csv(
    SELECTION_FILE
)

full_baseline = pd.read_csv(
    FULL_BASELINE_FILE
)


print(
    "Dataset shape:",
    X.shape,
)

print(
    "Subjects:",
    len(
        np.unique(
            subject_ids
        )
    ),
)

print(
    "ROCKET kernels:",
    N_KERNELS,
)


# ============================================================
# Storage
# ============================================================

fold_results = []

standard_subject_predictions = []
identity_subject_predictions = []


# ============================================================
# Final outer-fold evaluation
# ============================================================

for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    print(
        "\n\n"
        "============================================================"
    )

    print(
        f"OUTER FOLD {outer_fold}"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Get validation-selected parameters
    # --------------------------------------------------------

    fold_selection = (
        selection_table[
            selection_table[
                "outer_fold"
            ]
            == outer_fold
        ]
        .copy()
    )

    if len(
        fold_selection
    ) != 5:
        raise RuntimeError(
            f"Expected 5 lambda candidates "
            f"for fold {outer_fold}."
        )

    standard_row = (
        fold_selection[
            np.isclose(
                fold_selection[
                    "lambda"
                ],
                0.0,
            )
        ]
    )

    if len(
        standard_row
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "lambda=0 Standard SFD row."
        )

    standard_row = (
        standard_row.iloc[0]
    )

    selected_mask = (
        fold_selection[
            "selected"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    identity_rows = (
        fold_selection[
            selected_mask
        ]
    )

    if len(
        identity_rows
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one selected "
            f"identity candidate for fold "
            f"{outer_fold}."
        )

    identity_row = (
        identity_rows.iloc[0]
    )

    standard_percentage = float(
        standard_row[
            "retained_percentage"
        ]
    )

    selected_lambda = float(
        identity_row[
            "lambda"
        ]
    )

    identity_percentage = float(
        identity_row[
            "retained_percentage"
        ]
    )

    print(
        "Standard selected retention:",
        f"{standard_percentage:.3f}%",
    )

    print(
        "Identity selected lambda:",
        selected_lambda,
    )

    print(
        "Identity selected retention:",
        f"{identity_percentage:.3f}%",
    )

    # --------------------------------------------------------
    # Define development/test subjects
    # --------------------------------------------------------

    test_subjects = (
        fold_table.loc[
            fold_table[
                "fold"
            ]
            == outer_fold,
            "subject",
        ]
        .to_numpy()
    )

    development_subjects = (
        fold_table.loc[
            fold_table[
                "fold"
            ]
            != outer_fold,
            "subject",
        ]
        .to_numpy()
    )

    if (
        set(test_subjects)
        & set(development_subjects)
    ):
        raise RuntimeError(
            "Development/test subject leakage."
        )

    development_mask = np.isin(
        subject_ids,
        development_subjects,
    )

    test_mask = np.isin(
        subject_ids,
        test_subjects,
    )

    X_development = X[
        development_mask
    ]

    y_development = y_task[
        development_mask
    ]

    y_subject_development = y_subject[
        development_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y_task[
        test_mask
    ]

    y_subject_test = y_subject[
        test_mask
    ]

    test_subject_ids = subject_ids[
        test_mask
    ]

    print(
        "\nDevelopment subjects:",
        len(development_subjects),
    )

    print(
        "Test subjects:",
        len(test_subjects),
    )

    print(
        "Development windows:",
        len(X_development),
    )

    print(
        "Test windows:",
        len(X_test),
    )

    # ========================================================
    # STANDARD SFD
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "FITTING SUBJECT-SELECTED STANDARD SFD"
    )

    print(
        "----------------------------------------"
    )

    standard_rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    standard_model = DetachRocket(
        transformer=standard_rocket,
        set_percentage=standard_percentage,
        verbose=False,
    )

    start_time = time.time()

    standard_model.fit(
        X_development,
        y_development,
    )

    standard_fit_time = (
        time.time()
        - start_time
    )

    standard_actual_retained = (
        100
        * standard_model
        .retained_ratios_[
            standard_model
            .selected_step_index_
        ]
    )

    standard_features_retained = int(
        np.sum(
            standard_model
            .feature_mask_
        )
    )

    X_test_standard = (
        standard_model
        ._prepare_X(
            X_test
        )
    )

    # Window-level task score
    standard_window_predictions = (
        standard_model
        .classifier_
        .predict(
            X_test_standard
        )
    )

    standard_window_score = (
        balanced_accuracy_score(
            y_test,
            standard_window_predictions,
        )
    )

    # Subject-level task score
    standard_decision_scores = (
        standard_model
        .classifier_
        .decision_function(
            X_test_standard
        )
    )

    (
        standard_subject_score,
        standard_predictions,
    ) = evaluate_subject_level(
        standard_decision_scores,
        y_test,
        test_subject_ids,
    )

    standard_predictions.insert(
        0,
        "method",
        "standard_sfd",
    )

    standard_predictions.insert(
        0,
        "outer_fold",
        outer_fold,
    )

    standard_subject_predictions.append(
        standard_predictions
    )

    # Identity leakage
    (
        standard_identity_score,
        identity_chance,
    ) = identity_probe_accuracy(
        X_test_standard,
        y_subject_test,
        random_state=RANDOM_SEED,
    )

    print(
        "Actual retention:",
        f"{standard_actual_retained:.3f}%",
    )

    print(
        "Features retained:",
        standard_features_retained,
    )

    print(
        "Window task:",
        f"{standard_window_score * 100:.2f}%",
    )

    print(
        "Subject task:",
        f"{standard_subject_score * 100:.2f}%",
    )

    print(
        "Identity leakage:",
        f"{standard_identity_score * 100:.2f}%",
    )

    # ========================================================
    # IDENTITY-AWARE SFD
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "FITTING SUBJECT-SELECTED "
        "IDENTITY-AWARE SFD"
    )

    print(
        "----------------------------------------"
    )

    identity_rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    identity_model = DetachRocket(
        transformer=identity_rocket,
        set_percentage=identity_percentage,
        verbose=False,
    )

    start_time = time.time()

    identity_model.fit(
        X_development,
        y_development,
        y_identity=(
            y_subject_development
        ),
        identity_lambda=(
            selected_lambda
        ),
    )

    identity_fit_time = (
        time.time()
        - start_time
    )

    identity_actual_retained = (
        100
        * identity_model
        .retained_ratios_[
            identity_model
            .selected_step_index_
        ]
    )

    identity_features_retained = int(
        np.sum(
            identity_model
            .feature_mask_
        )
    )

    X_test_identity = (
        identity_model
        ._prepare_X(
            X_test
        )
    )

    # Window-level task score
    identity_window_predictions = (
        identity_model
        .classifier_
        .predict(
            X_test_identity
        )
    )

    identity_window_score = (
        balanced_accuracy_score(
            y_test,
            identity_window_predictions,
        )
    )

    # Subject-level task score
    identity_decision_scores = (
        identity_model
        .classifier_
        .decision_function(
            X_test_identity
        )
    )

    (
        identity_subject_score,
        identity_predictions,
    ) = evaluate_subject_level(
        identity_decision_scores,
        y_test,
        test_subject_ids,
    )

    identity_predictions.insert(
        0,
        "method",
        "identity_sfd",
    )

    identity_predictions.insert(
        0,
        "outer_fold",
        outer_fold,
    )

    identity_subject_predictions.append(
        identity_predictions
    )

    # Identity leakage
    (
        identity_identity_score,
        identity_chance_check,
    ) = identity_probe_accuracy(
        X_test_identity,
        y_subject_test,
        random_state=RANDOM_SEED,
    )

    if not np.isclose(
        identity_chance,
        identity_chance_check,
    ):
        raise RuntimeError(
            "Identity chance mismatch."
        )

    print(
        "Actual retention:",
        f"{identity_actual_retained:.3f}%",
    )

    print(
        "Features retained:",
        identity_features_retained,
    )

    print(
        "Window task:",
        f"{identity_window_score * 100:.2f}%",
    )

    print(
        "Subject task:",
        f"{identity_subject_score * 100:.2f}%",
    )

    print(
        "Identity leakage:",
        f"{identity_identity_score * 100:.2f}%",
    )

    print(
        "Identity chance:",
        f"{identity_chance * 100:.2f}%",
    )

    # ========================================================
    # Save fold
    # ========================================================

    fold_results.append(
        {
            "outer_fold": outer_fold,

            "standard_selected_percentage":
                standard_percentage,

            "standard_actual_retained_percentage":
                standard_actual_retained,

            "standard_retained_feature_count":
                standard_features_retained,

            "identity_lambda":
                selected_lambda,

            "identity_selected_percentage":
                identity_percentage,

            "identity_actual_retained_percentage":
                identity_actual_retained,

            "identity_retained_feature_count":
                identity_features_retained,

            "standard_window_balanced_accuracy":
                standard_window_score,

            "standard_subject_balanced_accuracy":
                standard_subject_score,

            "standard_identity_accuracy":
                standard_identity_score,

            "identity_window_balanced_accuracy":
                identity_window_score,

            "identity_subject_balanced_accuracy":
                identity_subject_score,

            "identity_identity_accuracy":
                identity_identity_score,

            "identity_chance":
                identity_chance,

            "standard_fit_time_seconds":
                standard_fit_time,

            "identity_fit_time_seconds":
                identity_fit_time,
        }
    )

    # --------------------------------------------------------
    # Memory cleanup
    # --------------------------------------------------------

    del standard_model
    del identity_model

    del standard_rocket
    del identity_rocket

    del X_test_standard
    del X_test_identity

    del X_development
    del X_test

    gc.collect()


# ============================================================
# DataFrames
# ============================================================

results = pd.DataFrame(
    fold_results
)

standard_predictions = pd.concat(
    standard_subject_predictions,
    ignore_index=True,
)

identity_predictions = pd.concat(
    identity_subject_predictions,
    ignore_index=True,
)

all_predictions = pd.concat(
    [
        standard_predictions,
        identity_predictions,
    ],
    ignore_index=True,
)


# ============================================================
# Integrity checks
# ============================================================

for method in [
    "standard_sfd",
    "identity_sfd",
]:

    method_predictions = (
        all_predictions[
            all_predictions[
                "method"
            ]
            == method
        ]
    )

    if len(
        method_predictions
    ) != 65:
        raise RuntimeError(
            f"{method}: expected 65 "
            f"subject predictions."
        )

    if not (
        method_predictions[
            "subject"
        ]
        .value_counts()
        == 1
    ).all():

        raise RuntimeError(
            f"{method}: each subject "
            f"must be tested exactly once."
        )


# ============================================================
# Pooled subject-level scores
# ============================================================

standard_pooled = (
    balanced_accuracy_score(
        standard_predictions[
            "true_label"
        ],
        standard_predictions[
            "predicted_label"
        ],
    )
)

identity_pooled = (
    balanced_accuracy_score(
        identity_predictions[
            "true_label"
        ],
        identity_predictions[
            "predicted_label"
        ],
    )
)


# ============================================================
# Save
# ============================================================

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    RESULT_FILE,
    index=False,
)

all_predictions.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# Helpers
# ============================================================

def mean_sd(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.std(
                values,
                ddof=1,
            )
        ),
    )


# ============================================================
# Load full ROCKET baseline
# ============================================================

full_task_mean, full_task_sd = mean_sd(
    full_baseline[
        "subject_balanced_accuracy"
    ]
)

full_identity_mean, full_identity_sd = mean_sd(
    full_baseline[
        "identity_accuracy"
    ]
)


# ============================================================
# New Standard SFD summary
# ============================================================

standard_task_mean, standard_task_sd = mean_sd(
    results[
        "standard_subject_balanced_accuracy"
    ]
)

standard_identity_mean, standard_identity_sd = mean_sd(
    results[
        "standard_identity_accuracy"
    ]
)

standard_retained_mean, standard_retained_sd = mean_sd(
    results[
        "standard_actual_retained_percentage"
    ]
)


# ============================================================
# New Identity-aware summary
# ============================================================

identity_task_mean, identity_task_sd = mean_sd(
    results[
        "identity_subject_balanced_accuracy"
    ]
)

identity_identity_mean, identity_identity_sd = mean_sd(
    results[
        "identity_identity_accuracy"
    ]
)

identity_retained_mean, identity_retained_sd = mean_sd(
    results[
        "identity_actual_retained_percentage"
    ]
)


# ============================================================
# Final output
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "FINAL SUBJECT-SELECTED OUTER-TEST RESULTS"
)

print(
    "============================================================"
)

print(
    "\nPer-fold results:"
)

print(
    results[
        [
            "outer_fold",
            "identity_lambda",
            "standard_actual_retained_percentage",
            "identity_actual_retained_percentage",
            "standard_subject_balanced_accuracy",
            "identity_subject_balanced_accuracy",
            "standard_identity_accuracy",
            "identity_identity_accuracy",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSUBJECT-LEVEL TASK BALANCED ACCURACY"
)

print(
    f"Full ROCKET: "
    f"{full_task_mean * 100:.2f}% "
    f"± {full_task_sd * 100:.2f}%"
)

print(
    f"Subject-selected Standard SFD: "
    f"{standard_task_mean * 100:.2f}% "
    f"± {standard_task_sd * 100:.2f}%"
)

print(
    f"Subject-selected Identity-aware SFD: "
    f"{identity_task_mean * 100:.2f}% "
    f"± {identity_task_sd * 100:.2f}%"
)


print(
    "\nPOOLED OUT-OF-FOLD SUBJECT BALANCED ACCURACY"
)

print(
    f"Standard SFD: "
    f"{standard_pooled * 100:.2f}%"
)

print(
    f"Identity-aware SFD: "
    f"{identity_pooled * 100:.2f}%"
)


print(
    "\nIDENTITY LEAKAGE"
)

print(
    f"Full ROCKET: "
    f"{full_identity_mean * 100:.2f}% "
    f"± {full_identity_sd * 100:.2f}%"
)

print(
    f"Subject-selected Standard SFD: "
    f"{standard_identity_mean * 100:.2f}% "
    f"± {standard_identity_sd * 100:.2f}%"
)

print(
    f"Subject-selected Identity-aware SFD: "
    f"{identity_identity_mean * 100:.2f}% "
    f"± {identity_identity_sd * 100:.2f}%"
)

print(
    f"Identity chance: "
    f"{results['identity_chance'].mean() * 100:.2f}%"
)


print(
    "\nFEATURE RETENTION"
)

print(
    f"Standard SFD: "
    f"{standard_retained_mean:.3f}% "
    f"± {standard_retained_sd:.3f}%"
)

print(
    f"Identity-aware SFD: "
    f"{identity_retained_mean:.3f}% "
    f"± {identity_retained_sd:.3f}%"
)


print(
    "\nSELECTED LAMBDAS"
)

print(
    results[
        "identity_lambda"
    ].tolist()
)


print(
    "\nSaved fold results to:"
)

print(
    RESULT_FILE
)

print(
    "\nSaved subject predictions to:"
)

print(
    PREDICTION_FILE
)