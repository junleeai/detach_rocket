"""
Identity-aware SFD evaluation on ds004504.

Lambda is selected using training + validation subjects only.

Selection rule:
1. Find the best validation task accuracy.
2. Keep lambdas within 5 percentage points of that best score.
3. Among eligible candidates, choose the lowest validation identity leakage.
4. Tie-break by higher task accuracy, then smaller lambda.

The selected lambda and retention percentage are then refit using all
non-test subjects and evaluated once on the untouched outer test fold.
"""

from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd

from aeon.transformations.collection.convolution_based import Rocket

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

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

STANDARD_FILE = Path(
    "results/thesis_rocket_real_standard_sfd.csv"
)

FULL_FILE = Path(
    "results/thesis_rocket_real_baseline.csv"
)

VALIDATION_FILE = Path(
    "results/thesis_rocket_real_identity_validation.csv"
)

RESULT_FILE = Path(
    "results/thesis_rocket_real_identity_sfd.csv"
)

PREDICTION_FILE = Path(
    "results/thesis_rocket_real_identity_sfd_subject_predictions.csv"
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
        subject_df["true_label"],
        subject_df["predicted_label"],
    )

    return score, subject_df


# ============================================================
# Identity probe
# ============================================================

def identity_probe_accuracy(
    features,
    subject_labels,
    random_state=42,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    true_labels = []
    predictions = []

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

        pred = probe.predict(
            features[
                probe_test_idx
            ]
        )

        true_labels.extend(
            subject_labels[
                probe_test_idx
            ]
        )

        predictions.extend(
            pred
        )

    return float(
        accuracy_score(
            true_labels,
            predictions,
        )
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

standard_results = pd.read_csv(
    STANDARD_FILE
)

full_results = pd.read_csv(
    FULL_FILE
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
    "Lambda candidates:",
    IDENTITY_LAMBDAS,
)

print(
    "Task tolerance:",
    f"{TASK_TOLERANCE * 100:.1f} percentage points",
)


# ============================================================
# Result storage
# ============================================================

validation_rows = []
final_rows = []
all_subject_predictions = []


# ============================================================
# Outer subject-wise folds
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
    # Subjects
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
            ~fold_table["fold"].isin(
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
            "Train/validation leakage."
        )

    if (
        set(train_subjects)
        & set(test_subjects)
    ):
        raise RuntimeError(
            "Train/test leakage."
        )

    if (
        set(val_subjects)
        & set(test_subjects)
    ):
        raise RuntimeError(
            "Validation/test leakage."
        )

    # --------------------------------------------------------
    # Masks
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
    # Arrays
    # --------------------------------------------------------

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

    y_subject_val = y_subject[
        val_mask
    ]

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
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )

    print(
        "Development:",
        X_development.shape,
    )

    print(
        "Test:",
        X_test.shape,
    )

    # ========================================================
    # PHASE 1
    # Compute ROCKET representation once
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 1: VALIDATION LAMBDA SELECTION"
    )

    print(
        "----------------------------------------"
    )

    print(
        "\nComputing training/validation ROCKET features..."
    )

    candidate_rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    # Fit ordinary Standard SFD first.
    # This gives us the exact transformed/scaled matrices,
    # full task alpha, and lambda=0 SFD curve.
    candidate_model = DetachRocket(
        transformer=candidate_rocket,
        trade_off=TRADE_OFF,
        verbose=False,
    )

    candidate_model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )

    X_train_features = (
        candidate_model
        .feature_matrix_
    )

    X_val_features = (
        candidate_model
        .feature_matrix_val_
    )

    task_alpha = (
        candidate_model
        .full_model_alpha_
    )

    # --------------------------------------------------------
    # Lambda = 0
    # --------------------------------------------------------

    candidate_results = []

    lambda0_step = (
        candidate_model
        .selected_step_index_
    )

    lambda0_mask = (
        candidate_model
        .importance_matrix_[
            lambda0_step
        ]
        > 0
    )

    lambda0_task = float(
        candidate_model
        .val_scores_[
            lambda0_step
        ]
    )

    lambda0_identity = (
        identity_probe_accuracy(
            X_val_features[
                :,
                lambda0_mask,
            ],
            y_subject_val,
            random_state=RANDOM_SEED,
        )
    )

    lambda0_retained = (
        100
        * candidate_model
        .retained_ratios_[
            lambda0_step
        ]
    )

    candidate_results.append(
        {
            "lambda": 0.0,
            "validation_task": (
                lambda0_task
            ),
            "validation_identity": (
                lambda0_identity
            ),
            "retained_percentage": (
                lambda0_retained
            ),
        }
    )

    # --------------------------------------------------------
    # Verify lambda=0 matches previous Standard SFD run
    # --------------------------------------------------------

    previous_standard = (
        standard_results.loc[
            standard_results[
                "outer_fold"
            ]
            == outer_fold
        ]
        .iloc[0]
    )

    previous_percentage = float(
        previous_standard[
            "selected_retained_percentage"
        ]
    )

    if not np.isclose(
        lambda0_retained,
        previous_percentage,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Fold {outer_fold}: lambda=0 retention "
            f"{lambda0_retained} does not match "
            f"previous Standard SFD retention "
            f"{previous_percentage}."
        )

    print(
        "\nlambda=0 consistency check: PASS"
    )

    # --------------------------------------------------------
    # Fit identity alpha once
    # --------------------------------------------------------

    print(
        "Selecting identity Ridge alpha..."
    )

    identity_cv = RidgeClassifierCV(
        alphas=ALPHAS
    )

    with quiet_ridge_warnings():

        identity_cv.fit(
            X_train_features,
            y_subject_train,
        )

    identity_alpha = (
        identity_cv.alpha_
    )

    print(
        "Identity alpha:",
        identity_alpha,
    )

    # --------------------------------------------------------
    # Evaluate positive lambdas
    # --------------------------------------------------------

    for identity_lambda in (
        IDENTITY_LAMBDAS[1:]
    ):

        print(
            f"\nTesting lambda={identity_lambda}..."
        )

        task_classifier = RidgeClassifier(
            alpha=task_alpha
        )

        identity_classifier = RidgeClassifier(
            alpha=identity_alpha
        )

        with quiet_ridge_warnings():

            identity_classifier.fit(
                X_train_features,
                y_subject_train,
            )

        start_time = time.time()

        (
            retained_ratios,
            train_scores,
            val_scores,
            importance_matrix,
        ) = feature_detachment(
            task_classifier,
            X_train_features,
            X_test=X_val_features,
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

        elapsed = (
            time.time()
            - start_time
        )

        (
            selected_step,
            _,
        ) = select_optimal_pruning(
            retained_ratios,
            val_scores,
            trade_off=TRADE_OFF,
        )

        validation_task = float(
            val_scores[
                selected_step
            ]
        )

        feature_mask = (
            importance_matrix[
                selected_step
            ]
            > 0
        )

        validation_identity = (
            identity_probe_accuracy(
                X_val_features[
                    :,
                    feature_mask,
                ],
                y_subject_val,
                random_state=RANDOM_SEED,
            )
        )

        retained_percentage = (
            100
            * retained_ratios[
                selected_step
            ]
        )

        candidate_results.append(
            {
                "lambda": (
                    identity_lambda
                ),
                "validation_task": (
                    validation_task
                ),
                "validation_identity": (
                    validation_identity
                ),
                "retained_percentage": (
                    retained_percentage
                ),
            }
        )

        print(
            "Validation task:",
            f"{validation_task * 100:.2f}%",
        )

        print(
            "Validation identity:",
            f"{validation_identity * 100:.2f}%",
        )

        print(
            "Features retained:",
            f"{retained_percentage:.3f}%",
        )

        print(
            "SFD time:",
            f"{elapsed:.1f} sec",
        )

    # ========================================================
    # Select lambda
    # ========================================================

    best_validation_task = max(
        result[
            "validation_task"
        ]
        for result
        in candidate_results
    )

    minimum_acceptable_task = max(
        0.0,
        best_validation_task
        - TASK_TOLERANCE,
    )

    eligible_candidates = [
        result
        for result
        in candidate_results
        if (
            result[
                "validation_task"
            ]
            >= minimum_acceptable_task
        )
    ]

    selected_candidate = sorted(
        eligible_candidates,
        key=lambda result: (
            result[
                "validation_identity"
            ],
            -result[
                "validation_task"
            ],
            result[
                "lambda"
            ],
        ),
    )[0]

    selected_lambda = (
        selected_candidate[
            "lambda"
        ]
    )

    selected_percentage = (
        selected_candidate[
            "retained_percentage"
        ]
    )

    # --------------------------------------------------------
    # Print validation table
    # --------------------------------------------------------

    print(
        "\nValidation selection"
    )

    print(
        "Lambda | Task | Identity | Retained | Eligible | Selected"
    )

    print(
        "-------|------|----------|----------|----------|---------"
    )

    for result in candidate_results:

        eligible = (
            result
            in eligible_candidates
        )

        selected = (
            result
            is selected_candidate
        )

        print(
            f"{result['lambda']:>6.2f} | "
            f"{100 * result['validation_task']:>5.2f}% | "
            f"{100 * result['validation_identity']:>7.2f}% | "
            f"{result['retained_percentage']:>7.3f}% | "
            f"{str(eligible):>8} | "
            f"{str(selected):>8}"
        )

        validation_rows.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "lambda": (
                    result[
                        "lambda"
                    ]
                ),
                "validation_task_accuracy": (
                    result[
                        "validation_task"
                    ]
                ),
                "validation_identity_accuracy": (
                    result[
                        "validation_identity"
                    ]
                ),
                "retained_percentage": (
                    result[
                        "retained_percentage"
                    ]
                ),
                "eligible": (
                    eligible
                ),
                "selected": (
                    selected
                ),
            }
        )

    print(
        "\nSelected lambda:",
        selected_lambda,
    )

    print(
        "Selected retention:",
        f"{selected_percentage:.3f}%",
    )

    # Free candidate-stage objects before final refit.
    del candidate_model
    del candidate_rocket
    del X_train_features
    del X_val_features
    del identity_cv

    gc.collect()

    # ========================================================
    # PHASE 2
    # Final identity-aware refit
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 2: FINAL IDENTITY-AWARE REFIT"
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
        set_percentage=selected_percentage,
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

    refit_time = (
        time.time()
        - start_time
    )

    actual_retained = (
        100
        * identity_model
        .retained_ratios_[
            identity_model
            .selected_step_index_
        ]
    )

    retained_features = int(
        np.sum(
            identity_model
            .feature_mask_
        )
    )

    print(
        "Actual retention:",
        f"{actual_retained:.3f}%",
    )

    print(
        "Retained features:",
        retained_features,
    )

    print(
        "Refit time:",
        f"{refit_time:.1f} sec",
    )

    # ========================================================
    # PHASE 3
    # Untouched outer test evaluation
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

    X_test_identity = (
        identity_model
        ._prepare_X(
            X_test
        )
    )

    # --------------------------------------------------------
    # Window task performance
    # --------------------------------------------------------

    window_predictions = (
        identity_model
        .classifier_
        .predict(
            X_test_identity
        )
    )

    window_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            window_predictions,
        )
    )

    # --------------------------------------------------------
    # Subject task performance
    # --------------------------------------------------------

    decision_scores = (
        identity_model
        .classifier_
        .decision_function(
            X_test_identity
        )
    )

    (
        subject_balanced_accuracy,
        subject_predictions,
    ) = evaluate_subject_level(
        decision_scores,
        y_test,
        test_subject_ids,
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

    identity_accuracy = (
        identity_probe_accuracy(
            X_test_identity,
            y_subject_test,
            random_state=RANDOM_SEED,
        )
    )

    identity_chance = (
        1.0
        / len(
            np.unique(
                y_subject_test
            )
        )
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        "Selected lambda:",
        selected_lambda,
    )

    print(
        "Window balanced accuracy:",
        f"{window_balanced_accuracy * 100:.2f}%",
    )

    print(
        "Subject balanced accuracy:",
        f"{subject_balanced_accuracy * 100:.2f}%",
    )

    print(
        "Identity-probe accuracy:",
        f"{identity_accuracy * 100:.2f}%",
    )

    print(
        "Identity chance:",
        f"{identity_chance * 100:.2f}%",
    )

    print(
        "Features retained:",
        f"{actual_retained:.3f}%",
    )

    final_rows.append(
        {
            "outer_fold": (
                outer_fold
            ),
            "selected_lambda": (
                selected_lambda
            ),
            "selected_retained_percentage": (
                selected_percentage
            ),
            "actual_retained_percentage": (
                actual_retained
            ),
            "retained_feature_count": (
                retained_features
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
            "refit_time_seconds": (
                refit_time
            ),
        }
    )

    del identity_model
    del identity_rocket
    del X_test_identity

    del X_train
    del X_val
    del X_development
    del X_test

    gc.collect()


# ============================================================
# Save results
# ============================================================

validation_df = pd.DataFrame(
    validation_rows
)

results = pd.DataFrame(
    final_rows
)

predictions = pd.concat(
    all_subject_predictions,
    ignore_index=True,
)

if len(
    predictions
) != 65:
    raise RuntimeError(
        "Expected exactly 65 "
        "out-of-fold subject predictions."
    )

if not (
    predictions[
        "subject"
    ]
    .value_counts()
    == 1
).all():
    raise RuntimeError(
        "A subject appears in the test set "
        "more or less than once."
    )

pooled_subject_accuracy = (
    balanced_accuracy_score(
        predictions[
            "true_label"
        ],
        predictions[
            "predicted_label"
        ],
    )
)

VALIDATION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False,
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
# Final comparison
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "IDENTITY-AWARE SFD REAL-DATA RESULTS"
)

print(
    "============================================================"
)

comparison_rows = []

for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    full_row = (
        full_results.loc[
            full_results[
                "outer_fold"
            ]
            == outer_fold
        ]
        .iloc[0]
    )

    standard_row = (
        standard_results.loc[
            standard_results[
                "outer_fold"
            ]
            == outer_fold
        ]
        .iloc[0]
    )

    identity_row = (
        results.loc[
            results[
                "outer_fold"
            ]
            == outer_fold
        ]
        .iloc[0]
    )

    comparison_rows.append(
        {
            "fold": outer_fold,
            "lambda": (
                identity_row[
                    "selected_lambda"
                ]
            ),
            "full_task": (
                full_row[
                    "subject_balanced_accuracy"
                ]
            ),
            "standard_task": (
                standard_row[
                    "subject_balanced_accuracy"
                ]
            ),
            "identity_task": (
                identity_row[
                    "subject_balanced_accuracy"
                ]
            ),
            "full_identity": (
                full_row[
                    "identity_accuracy"
                ]
            ),
            "standard_identity": (
                standard_row[
                    "identity_accuracy"
                ]
            ),
            "identity_identity": (
                identity_row[
                    "identity_accuracy"
                ]
            ),
            "standard_retained": (
                standard_row[
                    "actual_retained_percentage"
                ]
            ),
            "identity_retained": (
                identity_row[
                    "actual_retained_percentage"
                ]
            ),
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

print(
    "\nPer-fold comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)


def report_mean_sd(
    label,
    values,
    percentage=True,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    mean = np.mean(
        values
    )

    sd = np.std(
        values,
        ddof=1,
    )

    if percentage:

        print(
            f"{label}: "
            f"{mean * 100:.2f}% "
            f"± {sd * 100:.2f}%"
        )

    else:

        print(
            f"{label}: "
            f"{mean:.3f}% "
            f"± {sd:.3f}%"
        )


print(
    "\nSUBJECT-LEVEL TASK BALANCED ACCURACY"
)

report_mean_sd(
    "Full ROCKET",
    comparison[
        "full_task"
    ],
)

report_mean_sd(
    "Standard SFD",
    comparison[
        "standard_task"
    ],
)

report_mean_sd(
    "Identity-aware SFD",
    comparison[
        "identity_task"
    ],
)

print(
    "\nIDENTITY LEAKAGE"
)

report_mean_sd(
    "Full ROCKET",
    comparison[
        "full_identity"
    ],
)

report_mean_sd(
    "Standard SFD",
    comparison[
        "standard_identity"
    ],
)

report_mean_sd(
    "Identity-aware SFD",
    comparison[
        "identity_identity"
    ],
)

print(
    "\nFEATURE RETENTION"
)

report_mean_sd(
    "Standard SFD",
    comparison[
        "standard_retained"
    ],
    percentage=False,
)

report_mean_sd(
    "Identity-aware SFD",
    comparison[
        "identity_retained"
    ],
    percentage=False,
)

print(
    "\nSelected lambdas:"
)

print(
    results[
        "selected_lambda"
    ].tolist()
)

print(
    "\nPooled identity-aware "
    "out-of-fold subject balanced accuracy:"
)

print(
    f"{pooled_subject_accuracy * 100:.2f}%"
)

print(
    "\nIdentity chance level:"
)

print(
    f"{results['identity_chance'].mean() * 100:.2f}%"
)

print(
    "\nSaved validation results to:"
)

print(
    VALIDATION_FILE
)

print(
    "\nSaved final results to:"
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