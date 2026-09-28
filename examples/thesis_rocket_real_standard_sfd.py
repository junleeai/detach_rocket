"""
Standard SFD evaluation on the real ds004504 EEG dataset.

Protocol
--------
For each of five outer subject folds:

1. Three folds (39 subjects) are used for training.
2. One fold (13 subjects) is used for validation.
3. One fold (13 subjects) is held out for final testing.

The Standard SFD pruning level is selected using the existing
DetachRocket trade-off criterion (trade_off=0.1) on the training and
validation sets.

After the retention percentage is selected, Standard SFD is refit using
all non-test subjects (52 subjects).

The untouched outer test subjects are then evaluated for:

- window-level task balanced accuracy
- subject-level task balanced accuracy
- subject-identity leakage
- percentage of ROCKET features retained
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

RESULT_FILE = Path(
    "results/thesis_rocket_real_standard_sfd.csv"
)

PREDICTION_FILE = Path(
    "results/thesis_rocket_real_standard_sfd_subject_predictions.csv"
)

N_KERNELS = 10_000
N_FOLDS = 5
RANDOM_SEED = 42

TRADE_OFF = 0.1


# ============================================================
# Helper: subject-level task evaluation
# ============================================================

def evaluate_subject_level(
    decision_scores,
    y_true,
    subject_ids,
):
    """
    Average the 30 window-level Ridge decision scores for each
    subject and produce one AD/control prediction per subject.
    """

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

    score = (
        balanced_accuracy_score(
            subject_df[
                "true_label"
            ],
            subject_df[
                "predicted_label"
            ],
        )
    )

    return score, subject_df


# ============================================================
# Helper: identity leakage probe
# ============================================================

def identity_probe_accuracy(
    features,
    subject_labels,
    random_state=42,
):
    """
    Measure how accurately subject identity can be recovered
    from a frozen representation.
    """

    unique_subjects = np.unique(
        subject_labels
    )

    identity_chance = (
        1.0
        / len(unique_subjects)
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    identity_true = []
    identity_pred = []

    for (
        probe_train_idx,
        probe_test_idx,
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
                probe_train_idx
            ],
            subject_labels[
                probe_train_idx
            ],
        )

        predictions = probe.predict(
            features[
                probe_test_idx
            ]
        )

        identity_true.extend(
            subject_labels[
                probe_test_idx
            ]
        )

        identity_pred.extend(
            predictions
        )

    score = accuracy_score(
        identity_true,
        identity_pred,
    )

    return (
        float(score),
        float(identity_chance),
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

print(
    "Dataset shape:",
    X.shape,
)

print(
    "Subjects:",
    len(np.unique(subject_ids)),
)

print(
    "ROCKET kernels:",
    N_KERNELS,
)

print(
    "SFD trade-off:",
    TRADE_OFF,
)


# ============================================================
# Results
# ============================================================

fold_results = []
all_subject_predictions = []


# ============================================================
# Five outer subject-wise folds
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

    test_fold = outer_fold

    val_fold = (
        outer_fold
        % N_FOLDS
    ) + 1

    # --------------------------------------------------------
    # Subject sets
    # --------------------------------------------------------

    test_subjects = (
        fold_table.loc[
            fold_table["fold"]
            == test_fold,
            "subject",
        ]
        .to_numpy()
    )

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

    development_subjects = (
        np.concatenate(
            [
                train_subjects,
                val_subjects,
            ]
        )
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    if (
        set(train_subjects)
        & set(val_subjects)
    ):
        raise RuntimeError(
            "Train/validation subject leakage."
        )

    if (
        set(train_subjects)
        & set(test_subjects)
    ):
        raise RuntimeError(
            "Train/test subject leakage."
        )

    if (
        set(val_subjects)
        & set(test_subjects)
    ):
        raise RuntimeError(
            "Validation/test subject leakage."
        )

    print(
        "Train subjects:",
        len(train_subjects),
    )

    print(
        "Validation subjects:",
        len(val_subjects),
    )

    print(
        "Development subjects:",
        len(development_subjects),
    )

    print(
        "Test subjects:",
        len(test_subjects),
    )

    # --------------------------------------------------------
    # Window masks
    # --------------------------------------------------------

    train_mask = np.isin(
        subject_ids,
        train_subjects,
    )

    val_mask = np.isin(
        subject_ids,
        val_subjects,
    )

    development_mask = np.isin(
        subject_ids,
        development_subjects,
    )

    test_mask = np.isin(
        subject_ids,
        test_subjects,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X_train = X[
        train_mask
    ]

    y_train = y_task[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    y_val = y_task[
        val_mask
    ]

    X_development = X[
        development_mask
    ]

    y_development = y_task[
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
        "\nShapes:"
    )

    print(
        "  Train:",
        X_train.shape,
    )

    print(
        "  Validation:",
        X_val.shape,
    )

    print(
        "  Development:",
        X_development.shape,
    )

    print(
        "  Test:",
        X_test.shape,
    )

    # ========================================================
    # PHASE 1
    # Select Standard SFD retention using validation data
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 1: STANDARD SFD VALIDATION"
    )

    print(
        "----------------------------------------"
    )

    candidate_rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    candidate_model = DetachRocket(
        transformer=candidate_rocket,
        trade_off=TRADE_OFF,
        verbose=False,
    )

    start_time = time.time()

    candidate_model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )

    validation_time = (
        time.time()
        - start_time
    )

    selected_step = (
        candidate_model
        .selected_step_index_
    )

    selected_percentage = (
        100
        * candidate_model
        .retained_ratios_[
            selected_step
        ]
    )

    # This is ordinary WINDOW-LEVEL validation accuracy,
    # as used internally by the original SFD trade-off rule.
    validation_window_accuracy = float(
        candidate_model
        .val_scores_[
            selected_step
        ]
    )

    print(
        "Selected SFD step:",
        selected_step,
    )

    print(
        "Validation window accuracy:",
        f"{validation_window_accuracy * 100:.2f}%",
    )

    print(
        "Selected feature retention:",
        f"{selected_percentage:.2f}%",
    )

    print(
        "Validation fitting time:",
        f"{validation_time:.1f} seconds",
    )

    # We only need the selected percentage now.
    del candidate_model
    del candidate_rocket

    gc.collect()

    # ========================================================
    # PHASE 2
    # Refit Standard SFD using all non-test subjects
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 2: FINAL STANDARD SFD REFIT"
    )

    print(
        "----------------------------------------"
    )

    final_rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    standard_model = DetachRocket(
        transformer=final_rocket,
        set_percentage=selected_percentage,
        verbose=False,
    )

    start_time = time.time()

    standard_model.fit(
        X_development,
        y_development,
    )

    refit_time = (
        time.time()
        - start_time
    )

    actual_retained_percentage = (
        100
        * standard_model
        .retained_ratios_[
            standard_model
            .selected_step_index_
        ]
    )

    retained_feature_count = int(
        np.sum(
            standard_model
            .feature_mask_
        )
    )

    total_feature_count = int(
        len(
            standard_model
            .feature_mask_
        )
    )

    print(
        "Actual retained percentage:",
        f"{actual_retained_percentage:.2f}%",
    )

    print(
        "Features retained:",
        f"{retained_feature_count}"
        f" / {total_feature_count}",
    )

    print(
        "Final refit time:",
        f"{refit_time:.1f} seconds",
    )

    # ========================================================
    # PHASE 3
    # Evaluate untouched outer test subjects
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 3: OUTER TEST EVALUATION"
    )

    print(
        "----------------------------------------"
    )

    # Transform the test data ONCE into the final
    # Standard-SFD representation.
    X_test_standard = (
        standard_model
        ._prepare_X(
            X_test
        )
    )

    print(
        "Standard-SFD test feature shape:",
        X_test_standard.shape,
    )

    if np.isnan(
        X_test_standard
    ).any():
        raise RuntimeError(
            "NaNs found in Standard-SFD "
            "test features."
        )

    # --------------------------------------------------------
    # Window-level task performance
    # --------------------------------------------------------

    window_predictions = (
        standard_model
        .classifier_
        .predict(
            X_test_standard
        )
    )

    window_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            window_predictions,
        )
    )

    # --------------------------------------------------------
    # Subject-level task performance
    # --------------------------------------------------------

    decision_scores = (
        standard_model
        .classifier_
        .decision_function(
            X_test_standard
        )
    )

    (
        subject_balanced_accuracy,
        subject_predictions,
    ) = evaluate_subject_level(
        decision_scores=decision_scores,
        y_true=y_test,
        subject_ids=test_subject_ids,
    )

    subject_predictions.insert(
        0,
        "outer_fold",
        outer_fold,
    )

    all_subject_predictions.append(
        subject_predictions
    )

    # --------------------------------------------------------
    # Identity leakage
    # --------------------------------------------------------

    (
        identity_accuracy,
        identity_chance,
    ) = identity_probe_accuracy(
        X_test_standard,
        y_subject_test,
        random_state=RANDOM_SEED,
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        "Window-level balanced accuracy:",
        f"{window_balanced_accuracy * 100:.2f}%",
    )

    print(
        "Subject-level balanced accuracy:",
        f"{subject_balanced_accuracy * 100:.2f}%",
    )

    print(
        "Identity-probe accuracy:",
        f"{identity_accuracy * 100:.2f}%",
    )

    print(
        "Identity chance level:",
        f"{identity_chance * 100:.2f}%",
    )

    print(
        "Features retained:",
        f"{actual_retained_percentage:.2f}%",
    )

    # --------------------------------------------------------
    # Save fold result
    # --------------------------------------------------------

    fold_results.append(
        {
            "outer_fold": outer_fold,
            "train_subjects": len(
                train_subjects
            ),
            "validation_subjects": len(
                val_subjects
            ),
            "development_subjects": len(
                development_subjects
            ),
            "test_subjects": len(
                test_subjects
            ),
            "selected_step": (
                selected_step
            ),
            "validation_window_accuracy": (
                validation_window_accuracy
            ),
            "selected_retained_percentage": (
                selected_percentage
            ),
            "actual_retained_percentage": (
                actual_retained_percentage
            ),
            "retained_feature_count": (
                retained_feature_count
            ),
            "total_feature_count": (
                total_feature_count
            ),
            "window_balanced_accuracy": (
                window_balanced_accuracy
            ),
            "subject_balanced_accuracy": (
                subject_balanced_accuracy
            ),
            "identity_accuracy": (
                identity_accuracy
            ),
            "identity_chance": (
                identity_chance
            ),
            "validation_time_seconds": (
                validation_time
            ),
            "refit_time_seconds": (
                refit_time
            ),
        }
    )

    # --------------------------------------------------------
    # Free memory before next outer fold
    # --------------------------------------------------------

    del standard_model
    del final_rocket

    del X_train
    del X_val
    del X_development
    del X_test
    del X_test_standard

    gc.collect()


# ============================================================
# Aggregate results
# ============================================================

results = pd.DataFrame(
    fold_results
)

predictions = pd.concat(
    all_subject_predictions,
    ignore_index=True,
)

# Every subject must appear exactly once in the outer test sets.
subject_test_counts = (
    predictions[
        "subject"
    ]
    .value_counts()
)

if not (
    subject_test_counts == 1
).all():
    raise RuntimeError(
        "Some subjects were tested more "
        "or less than once."
    )

if len(
    predictions
) != 65:
    raise RuntimeError(
        f"Expected 65 subject predictions, "
        f"found {len(predictions)}."
    )


# ============================================================
# Pooled out-of-fold subject task score
# ============================================================

pooled_subject_balanced_accuracy = (
    balanced_accuracy_score(
        predictions[
            "true_label"
        ],
        predictions[
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

predictions.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "STANDARD SFD REAL-DATA RESULTS"
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
            "selected_retained_percentage",
            "actual_retained_percentage",
            "window_balanced_accuracy",
            "subject_balanced_accuracy",
            "identity_accuracy",
        ]
    ].to_string(
        index=False
    )
)


def mean_sd(
    column,
):
    values = results[
        column
    ].to_numpy()

    return (
        np.mean(values),
        np.std(
            values,
            ddof=1,
        ),
    )


window_mean, window_sd = mean_sd(
    "window_balanced_accuracy"
)

subject_mean, subject_sd = mean_sd(
    "subject_balanced_accuracy"
)

identity_mean, identity_sd = mean_sd(
    "identity_accuracy"
)

retained_mean, retained_sd = mean_sd(
    "actual_retained_percentage"
)

identity_chance = results[
    "identity_chance"
].mean()


print(
    "\nWindow-level balanced accuracy:"
)

print(
    f"{window_mean * 100:.2f}% "
    f"± {window_sd * 100:.2f}%"
)

print(
    "\nSubject-level balanced accuracy:"
)

print(
    f"{subject_mean * 100:.2f}% "
    f"± {subject_sd * 100:.2f}%"
)

print(
    "\nPooled out-of-fold "
    "subject balanced accuracy:"
)

print(
    f"{pooled_subject_balanced_accuracy * 100:.2f}%"
)

print(
    "\nIdentity-probe accuracy:"
)

print(
    f"{identity_mean * 100:.2f}% "
    f"± {identity_sd * 100:.2f}%"
)

print(
    "\nIdentity chance level:"
)

print(
    f"{identity_chance * 100:.2f}%"
)

print(
    "\nFeatures retained:"
)

print(
    f"{retained_mean:.2f}% "
    f"± {retained_sd:.2f}%"
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