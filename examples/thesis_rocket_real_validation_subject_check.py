"""
Diagnostic check of subject-level validation performance.

IMPORTANT:
- This script NEVER evaluates the outer test subjects.
- It does NOT overwrite the original experiment.
- It reproduces each lambda candidate's validation-selected SFD
  representation and measures task performance at the SUBJECT level.

Purpose:
Compare the original window-level validation criterion with the
subject-level balanced-accuracy criterion used as the primary
real-data task metric.
"""

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from aeon.transformations.collection.convolution_based import Rocket

from sklearn.linear_model import (
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.metrics import balanced_accuracy_score

from detach_rocket.detach_classes import DetachRocket
from detach_rocket.model_selection import select_optimal_pruning
from detach_rocket.sfd import feature_detachment
from detach_rocket._warnings import quiet_ridge_warnings


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

ORIGINAL_VALIDATION_FILE = Path(
    "results/thesis_rocket_real_identity_validation.csv"
)

OUTPUT_FILE = Path(
    "results/"
    "thesis_rocket_real_validation_subject_check.csv"
)

N_KERNELS = 10_000
N_FOLDS = 5
RANDOM_SEED = 42

TRADE_OFF = 0.1
TASK_TOLERANCE = 0.05

IDENTITY_LAMBDAS = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
]

ALPHAS = np.logspace(
    -10,
    10,
    20,
)


# ============================================================
# Subject-level validation score
# ============================================================

def subject_level_balanced_accuracy(
    decision_scores,
    y_true,
    subject_ids,
):
    """
    Average window-level Ridge decision scores within each subject,
    then calculate balanced accuracy across subjects.
    """

    true_subject_labels = []
    predicted_subject_labels = []

    for subject_id in np.unique(
        subject_ids
    ):

        mask = (
            subject_ids
            == subject_id
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

        true_subject_labels.append(
            true_label
        )

        predicted_subject_labels.append(
            predicted_label
        )

    return balanced_accuracy_score(
        true_subject_labels,
        predicted_subject_labels,
    )


# ============================================================
# Fit training-only classifier on a selected feature mask
# ============================================================

def evaluate_selected_representation(
    X_train_features,
    y_train,
    X_val_features,
    y_val,
    val_subject_ids,
    feature_mask,
):
    """
    Fit the final task classifier using ONLY the training subjects.

    Validation subjects are used only for evaluation.
    """

    if np.sum(feature_mask) == 0:
        raise RuntimeError(
            "Selected feature mask contains zero features."
        )

    X_train_selected = (
        X_train_features[
            :,
            feature_mask,
        ]
    )

    X_val_selected = (
        X_val_features[
            :,
            feature_mask,
        ]
    )

    classifier = RidgeClassifierCV(
        alphas=ALPHAS
    )

    with quiet_ridge_warnings():
        classifier.fit(
            X_train_selected,
            y_train,
        )

    # Window-level validation accuracy
    window_predictions = (
        classifier.predict(
            X_val_selected
        )
    )

    window_balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            window_predictions,
        )
    )

    # Subject-level validation accuracy
    decision_scores = (
        classifier.decision_function(
            X_val_selected
        )
    )

    subject_balanced_accuracy = (
        subject_level_balanced_accuracy(
            decision_scores,
            y_val,
            val_subject_ids,
        )
    )

    return (
        float(window_balanced_accuracy),
        float(subject_balanced_accuracy),
        float(classifier.alpha_),
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

original_validation = pd.read_csv(
    ORIGINAL_VALIDATION_FILE
)

print(
    "Dataset:",
    X.shape,
)

print(
    "Subjects:",
    len(np.unique(subject_ids)),
)

print(
    "This script does NOT evaluate outer test data."
)


# ============================================================
# Results
# ============================================================

rows = []


# ============================================================
# Five outer configurations
# ============================================================

for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    print(
        "\n"
        "============================================================"
    )

    print(
        f"FOLD {outer_fold}"
    )

    print(
        "============================================================"
    )

    test_fold = outer_fold

    val_fold = (
        outer_fold
        % N_FOLDS
    ) + 1

    # --------------------------------------------------------
    # Subject sets
    # --------------------------------------------------------

    val_subjects = (
        fold_table.loc[
            fold_table["fold"]
            == val_fold,
            "subject",
        ]
        .to_numpy()
    )

    train_subjects = (
        fold_table.loc[
            ~fold_table[
                "fold"
            ].isin(
                [
                    test_fold,
                    val_fold,
                ]
            ),
            "subject",
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Critically, test subjects are NEVER loaded into
    # candidate evaluation.
    # --------------------------------------------------------

    train_mask = np.isin(
        subject_ids,
        train_subjects,
    )

    val_mask = np.isin(
        subject_ids,
        val_subjects,
    )

    X_train = X[
        train_mask
    ]

    y_train = y_task[
        train_mask
    ]

    y_subject_train = y_subject[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    y_val = y_task[
        val_mask
    ]

    val_subject_ids = subject_ids[
        val_mask
    ]

    print(
        "Train subjects:",
        len(train_subjects),
    )

    print(
        "Validation subjects:",
        len(val_subjects),
    )

    print(
        "Train windows:",
        len(X_train),
    )

    print(
        "Validation windows:",
        len(X_val),
    )

    # ========================================================
    # Generate ROCKET features once
    # ========================================================

    print(
        "\nComputing ROCKET + lambda=0 SFD..."
    )

    rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    standard_model = DetachRocket(
        transformer=rocket,
        trade_off=TRADE_OFF,
        verbose=False,
    )

    standard_model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )

    X_train_features = (
        standard_model
        .feature_matrix_
    )

    X_val_features = (
        standard_model
        .feature_matrix_val_
    )

    task_alpha = (
        standard_model
        .full_model_alpha_
    )

    # --------------------------------------------------------
    # Fit identity alpha once for this training fold
    # --------------------------------------------------------

    identity_cv = RidgeClassifierCV(
        alphas=ALPHAS
    )

    with quiet_ridge_warnings():
        identity_cv.fit(
            X_train_features,
            y_subject_train,
        )

    identity_alpha = float(
        identity_cv.alpha_
    )

    print(
        "Task alpha:",
        task_alpha,
    )

    print(
        "Identity alpha:",
        identity_alpha,
    )

    # ========================================================
    # Evaluate every lambda
    # ========================================================

    for identity_lambda in (
        IDENTITY_LAMBDAS
    ):

        print(
            f"\nLambda = {identity_lambda}"
        )

        # ----------------------------------------------------
        # lambda = 0 already exists from Standard SFD
        # ----------------------------------------------------

        if identity_lambda == 0.0:

            retained_ratios = (
                standard_model
                .retained_ratios_
            )

            val_scores = (
                standard_model
                .val_scores_
            )

            importance_matrix = (
                standard_model
                .importance_matrix_
            )

            selected_step = (
                standard_model
                .selected_step_index_
            )

        # ----------------------------------------------------
        # Positive identity penalty
        # ----------------------------------------------------

        else:

            task_classifier = (
                RidgeClassifier(
                    alpha=task_alpha
                )
            )

            identity_classifier = (
                RidgeClassifier(
                    alpha=identity_alpha
                )
            )

            with quiet_ridge_warnings():
                identity_classifier.fit(
                    X_train_features,
                    y_subject_train,
                )

            (
                retained_ratios,
                train_scores,
                val_scores,
                importance_matrix,
            ) = feature_detachment(
                task_classifier,
                X_train_features,
                X_test=(
                    X_val_features
                ),
                y_train=y_train,
                y_test=y_val,
                verbose=False,
                multiclass_type="max",
                identity_classifier=(
                    identity_classifier
                ),
                y_identity=(
                    y_subject_train
                ),
                identity_lambda=(
                    identity_lambda
                ),
            )

            (
                selected_step,
                _,
            ) = select_optimal_pruning(
                retained_ratios,
                val_scores,
                trade_off=TRADE_OFF,
            )

        # ----------------------------------------------------
        # Representation chosen by ORIGINAL SFD criterion
        # ----------------------------------------------------

        feature_mask = (
            importance_matrix[
                selected_step
            ]
            > 0
        )

        retained_percentage = (
            100
            * retained_ratios[
                selected_step
            ]
        )

        original_window_accuracy = float(
            val_scores[
                selected_step
            ]
        )

        # ----------------------------------------------------
        # NEW diagnostic:
        # train-only classifier, evaluated at subject level
        # ----------------------------------------------------

        (
            diagnostic_window_balanced_accuracy,
            subject_balanced_accuracy,
            diagnostic_alpha,
        ) = evaluate_selected_representation(
            X_train_features=(
                X_train_features
            ),
            y_train=y_train,
            X_val_features=(
                X_val_features
            ),
            y_val=y_val,
            val_subject_ids=(
                val_subject_ids
            ),
            feature_mask=(
                feature_mask
            ),
        )

        # ----------------------------------------------------
        # Retrieve the already-recorded identity leakage
        # from the original experiment.
        # ----------------------------------------------------

        previous_row = (
            original_validation[
                (
                    original_validation[
                        "outer_fold"
                    ]
                    == outer_fold
                )
                &
                np.isclose(
                    original_validation[
                        "lambda"
                    ],
                    identity_lambda,
                )
            ]
        )

        if len(previous_row) != 1:
            raise RuntimeError(
                "Could not uniquely match "
                f"fold {outer_fold}, "
                f"lambda {identity_lambda}."
            )

        previous_row = (
            previous_row.iloc[0]
        )

        validation_identity = float(
            previous_row[
                "validation_identity_accuracy"
            ]
        )

        # Reproducibility checks
        if not np.isclose(
            retained_percentage,
            float(
                previous_row[
                    "retained_percentage"
                ]
            ),
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Retention mismatch in fold "
                f"{outer_fold}, lambda "
                f"{identity_lambda}."
            )

        if not np.isclose(
            original_window_accuracy,
            float(
                previous_row[
                    "validation_task_accuracy"
                ]
            ),
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Validation-score mismatch in "
                f"fold {outer_fold}, lambda "
                f"{identity_lambda}."
            )

        rows.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "lambda": (
                    identity_lambda
                ),
                "selected_step": (
                    selected_step
                ),
                "retained_percentage": (
                    retained_percentage
                ),
                "original_window_accuracy": (
                    original_window_accuracy
                ),
                "diagnostic_window_balanced_accuracy": (
                    diagnostic_window_balanced_accuracy
                ),
                "subject_balanced_accuracy": (
                    subject_balanced_accuracy
                ),
                "validation_identity_accuracy": (
                    validation_identity
                ),
                "diagnostic_task_alpha": (
                    diagnostic_alpha
                ),
                "original_eligible": bool(
                    previous_row[
                        "eligible"
                    ]
                ),
                "original_selected": bool(
                    previous_row[
                        "selected"
                    ]
                ),
            }
        )

        print(
            "  Original window accuracy:",
            f"{original_window_accuracy * 100:.2f}%",
        )

        print(
            "  Subject balanced accuracy:",
            f"{subject_balanced_accuracy * 100:.2f}%",
        )

        print(
            "  Identity leakage:",
            f"{validation_identity * 100:.2f}%",
        )

        print(
            "  Retained:",
            f"{retained_percentage:.3f}%",
        )

    # --------------------------------------------------------
    # Free this fold
    # --------------------------------------------------------

    del standard_model
    del rocket
    del identity_cv

    del X_train
    del X_val
    del X_train_features
    del X_val_features

    gc.collect()


# ============================================================
# Determine what SUBJECT-LEVEL lambda rule would choose
# ============================================================

results = pd.DataFrame(
    rows
)

results[
    "subject_metric_eligible"
] = False

results[
    "subject_metric_selected"
] = False


for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    fold_mask = (
        results[
            "outer_fold"
        ]
        == outer_fold
    )

    fold_results = (
        results.loc[
            fold_mask
        ]
        .copy()
    )

    best_subject_task = (
        fold_results[
            "subject_balanced_accuracy"
        ]
        .max()
    )

    minimum_acceptable = max(
        0.0,
        best_subject_task
        - TASK_TOLERANCE,
    )

    eligible_mask = (
        fold_results[
            "subject_balanced_accuracy"
        ]
        >= minimum_acceptable
    )

    eligible = (
        fold_results.loc[
            eligible_mask
        ]
        .copy()
    )

    selected_index = (
        eligible.sort_values(
            by=[
                "validation_identity_accuracy",
                "subject_balanced_accuracy",
                "lambda",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .index[0]
    )

    results.loc[
        eligible.index,
        "subject_metric_eligible",
    ] = True

    results.loc[
        selected_index,
        "subject_metric_selected",
    ] = True


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Print comparison
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "SUBJECT-LEVEL VALIDATION CHECK"
)

print(
    "============================================================"
)


for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    fold_results = results[
        results[
            "outer_fold"
        ]
        == outer_fold
    ]

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        f"FOLD {outer_fold}"
    )

    print(
        "----------------------------------------"
    )

    print(
        fold_results[
            [
                "lambda",
                "original_window_accuracy",
                "subject_balanced_accuracy",
                "validation_identity_accuracy",
                "retained_percentage",
                "original_selected",
                "subject_metric_selected",
            ]
        ].to_string(
            index=False
        )
    )

    original_lambda = float(
        fold_results.loc[
            fold_results[
                "original_selected"
            ],
            "lambda",
        ].iloc[0]
    )

    subject_lambda = float(
        fold_results.loc[
            fold_results[
                "subject_metric_selected"
            ],
            "lambda",
        ].iloc[0]
    )

    print(
        "\nOriginal selected lambda:",
        original_lambda,
    )

    print(
        "Subject-metric selected lambda:",
        subject_lambda,
    )


print(
    "\n============================================================"
)

print(
    "SUMMARY"
)

print(
    "============================================================"
)

original_lambdas = (
    results.loc[
        results[
            "original_selected"
        ]
    ]
    .sort_values(
        "outer_fold"
    )[
        "lambda"
    ]
    .tolist()
)

subject_lambdas = (
    results.loc[
        results[
            "subject_metric_selected"
        ]
    ]
    .sort_values(
        "outer_fold"
    )[
        "lambda"
    ]
    .tolist()
)

print(
    "\nOriginal window-metric lambdas:"
)

print(
    original_lambdas
)

print(
    "\nSubject-metric lambdas:"
)

print(
    subject_lambdas
)

print(
    "\nSaved diagnostic results to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "No outer-test predictions were generated "
    "by this script."
)