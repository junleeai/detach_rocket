"""Validation-only lambda selection for identity-aware SFD.

This experiment uses subject-wise train / validation / test partitions.

For each outer test fold:

1. Three subject folds are used for training.
2. One subject fold is used for validation.
3. One subject fold is held out as the final test set.

Candidate identity penalties are evaluated using only training and validation
subjects.

Lambda selection rule
---------------------
1. Find the best validation task performance among all lambda candidates.
2. Candidate lambdas must have validation task performance no more than
   5 percentage points below the best validation task performance.
3. Among eligible lambdas, choose the one with the lowest validation
   identity-probe accuracy. 
4. Ties are broken by:
      a. higher validation task accuracy
      b. smaller lambda

After lambda and feature-retention percentage are selected, the final model
is refit using all non-test subjects.

The outer test subjects are evaluated only after model selection is complete.
"""

import csv
import os
import time

import numpy as np

from aeon.transformations.collection.convolution_based import Rocket
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from detach_rocket.detach_classes import DetachRocket
from thesis_synthetic_data import generate_synthetic_datasets


# =========================================================
# Experimental settings
# =========================================================

RANDOM_SEED = 42

TASK_AMPLITUDE = 0.50
IDENTITY_CONDITION = "strong"

N_KERNELS = 10_000
N_OUTER_FOLDS = 5

IDENTITY_LAMBDAS = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
]

# Maximum allowed validation task-performance loss relative
# to the best-performing lambda candidate.
TASK_TOLERANCE = 0.05

RESULTS_DIR = "results"

FINAL_RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_nested_results.csv",
)

VALIDATION_RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_nested_validation.csv",
)


# =========================================================
# Generate synthetic data
# =========================================================

(
    datasets,
    y_task,
    y_subject,
    subject_classes,
    identity_frequencies,
) = generate_synthetic_datasets(
    task_amplitude=TASK_AMPLITUDE,
)

X = datasets[IDENTITY_CONDITION]


print("\nExperiment settings:")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Task amplitude: {TASK_AMPLITUDE}")
print(f"  Identity condition: {IDENTITY_CONDITION}")
print(f"  ROCKET kernels: {N_KERNELS}")
print(f"  Lambda candidates: {IDENTITY_LAMBDAS}")
print(
    f"  Task-performance tolerance: "
    f"{100 * TASK_TOLERANCE:.1f} percentage points"
)


# =========================================================
# Create balanced subject-wise folds
# =========================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

class_0_subjects = np.where(
    subject_classes == 0
)[0]

class_1_subjects = np.where(
    subject_classes == 1
)[0]

rng.shuffle(
    class_0_subjects
)

rng.shuffle(
    class_1_subjects
)

subject_folds = []

for fold_index in range(
    N_OUTER_FOLDS
):
    fold = np.concatenate(
        [
            class_0_subjects[
                fold_index * 2:
                fold_index * 2 + 2
            ],
            class_1_subjects[
                fold_index * 2:
                fold_index * 2 + 2
            ],
        ]
    )

    subject_folds.append(
        fold
    )


print("\nSubject-wise folds:")

for i, fold in enumerate(
    subject_folds
):
    print(
        f"Fold {i + 1}:",
        [
            f"S{s + 1:02d}"
            for s in fold
        ],
        "| classes:",
        subject_classes[fold],
    )


# =========================================================
# Identity-leakage probe
# =========================================================


def identity_probe_accuracy(
    features,
    subject_labels,
    random_state=42,
):
    """Measure subject-identity information in a frozen representation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    scores = []

    for (
        probe_train_idx,
        probe_test_idx,
    ) in cv.split(
        features,
        subject_labels,
    ):
        X_probe_train = features[
            probe_train_idx
        ]

        X_probe_test = features[
            probe_test_idx
        ]

        y_probe_train = subject_labels[
            probe_train_idx
        ]

        y_probe_test = subject_labels[
            probe_test_idx
        ]

        probe = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )

        probe.fit(
            X_probe_train,
            y_probe_train,
        )

        score = probe.score(
            X_probe_test,
            y_probe_test,
        )

        scores.append(
            score
        )

    return float(
        np.mean(scores)
    )


# =========================================================
# Utility function
# =========================================================


def add_channel_dimension(
    X_data,
):
    """Convert (recordings, timepoints) to ROCKET 3D input."""

    return X_data[
        :,
        np.newaxis,
        :
    ]


# =========================================================
# Store results
# =========================================================

validation_rows = []
final_rows = []


# =========================================================
# Outer subject-wise evaluation
# =========================================================

for test_fold_index in range(
    N_OUTER_FOLDS
):

    print(
        "\n\n"
        "############################################################"
    )

    print(
        f"OUTER TEST FOLD "
        f"{test_fold_index + 1}"
    )

    print(
        "############################################################"
    )

    # -----------------------------------------------------
    # Define validation fold
    # -----------------------------------------------------

    val_fold_index = (
        test_fold_index + 1
    ) % N_OUTER_FOLDS

    train_fold_indices = [
        i
        for i in range(
            N_OUTER_FOLDS
        )
        if i not in [
            test_fold_index,
            val_fold_index,
        ]
    ]

    test_subjects = subject_folds[
        test_fold_index
    ]

    val_subjects = subject_folds[
        val_fold_index
    ]

    train_subjects = np.concatenate(
        [
            subject_folds[i]
            for i in train_fold_indices
        ]
    )

    # All non-test subjects are used after
    # hyperparameter selection.
    development_subjects = np.concatenate(
        [
            train_subjects,
            val_subjects,
        ]
    )

    print(
        "\nTraining subjects:",
        [
            f"S{s + 1:02d}"
            for s in train_subjects
        ],
    )

    print(
        "Validation subjects:",
        [
            f"S{s + 1:02d}"
            for s in val_subjects
        ],
    )

    print(
        "Final test subjects:",
        [
            f"S{s + 1:02d}"
            for s in test_subjects
        ],
    )

    # -----------------------------------------------------
    # Recording-level masks
    # -----------------------------------------------------

    train_mask = np.isin(
        y_subject,
        train_subjects,
    )

    val_mask = np.isin(
        y_subject,
        val_subjects,
    )

    test_mask = np.isin(
        y_subject,
        test_subjects,
    )

    development_mask = np.isin(
        y_subject,
        development_subjects,
    )

    # -----------------------------------------------------
    # Training data
    # -----------------------------------------------------

    X_train = add_channel_dimension(
        X[train_mask]
    )

    y_train = y_task[
        train_mask
    ]

    y_subject_train = y_subject[
        train_mask
    ]

    # -----------------------------------------------------
    # Validation data
    # -----------------------------------------------------

    X_val = add_channel_dimension(
        X[val_mask]
    )

    y_val = y_task[
        val_mask
    ]

    y_subject_val = y_subject[
        val_mask
    ]

    # -----------------------------------------------------
    # Development data
    # -----------------------------------------------------

    X_development = add_channel_dimension(
        X[development_mask]
    )

    y_development = y_task[
        development_mask
    ]

    y_subject_development = y_subject[
        development_mask
    ]

    # -----------------------------------------------------
    # Final test data
    # -----------------------------------------------------

    X_test = add_channel_dimension(
        X[test_mask]
    )

    y_test = y_task[
        test_mask
    ]

    y_subject_test = y_subject[
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

    # =====================================================
    # PHASE 1
    # Validation-only lambda evaluation
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 1: VALIDATION-ONLY LAMBDA SELECTION"
    )

    print(
        "----------------------------------------"
    )

    candidate_results = []

    for identity_lambda in (
        IDENTITY_LAMBDAS
    ):

        print(
            f"\nTesting lambda = "
            f"{identity_lambda}"
        )

        rocket = Rocket(
            n_kernels=N_KERNELS,
            random_state=RANDOM_SEED,
        )

        candidate_model = DetachRocket(
            transformer=rocket,
            trade_off=0.1,
            verbose=False,
        )

        start_time = time.time()

        candidate_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            y_identity=y_subject_train,
            identity_lambda=identity_lambda,
        )

        elapsed = (
            time.time()
            - start_time
        )

        # -------------------------------------------------
        # Validation task performance
        # -------------------------------------------------
        #
        # val_scores_ contains the validation accuracy
        # measured DURING SFD, before the final classifier
        # is retrained using train + validation data.
        #
        # The synthetic validation folds are exactly class
        # balanced, so ordinary accuracy equals balanced
        # accuracy here.

        selected_step = (
            candidate_model
            .selected_step_index_
        )

        validation_task_score = float(
            candidate_model.val_scores_[
                selected_step
            ]
        )

        # -------------------------------------------------
        # Selected representation
        # -------------------------------------------------

        validation_features = (
            candidate_model._prepare_X(
                X_val
            )
        )

        # -------------------------------------------------
        # Validation identity leakage
        # -------------------------------------------------

        validation_identity_score = (
            identity_probe_accuracy(
                validation_features,
                y_subject_val,
                random_state=RANDOM_SEED,
            )
        )

        # -------------------------------------------------
        # Selected feature-retention percentage
        # -------------------------------------------------

        retained_percentage = (
            100
            * candidate_model
            .retained_ratios_[
                selected_step
            ]
        )

        candidate_result = {
            "lambda": (
                identity_lambda
            ),
            "validation_task": (
                validation_task_score
            ),
            "validation_identity": (
                validation_identity_score
            ),
            "retained_percentage": (
                retained_percentage
            ),
            "time_seconds": (
                elapsed
            ),
        }

        candidate_results.append(
            candidate_result
        )

        print(
            f"  Validation task: "
            f"{100 * validation_task_score:.2f}%"
        )

        print(
            f"  Validation identity: "
            f"{100 * validation_identity_score:.2f}%"
        )

        print(
            f"  Features retained: "
            f"{retained_percentage:.2f}%"
        )

    # =====================================================
    # Determine eligible lambda values
    # =====================================================

        # Keep the lambda=0 candidate because its selected
    # retention percentage is still needed later for
    # the final Standard SFD baseline.
    standard_candidate = next(
        result
        for result in candidate_results
        if result["lambda"] == 0.0
    )

    # Find the best validation task performance across
    # all lambda candidates.
    best_validation_task = max(
        result["validation_task"]
        for result in candidate_results
    )

    # A lambda is eligible only if its validation task
    # performance is within TASK_TOLERANCE of the best
    # candidate.
    minimum_acceptable_task = max(
        0.0,
        best_validation_task
        - TASK_TOLERANCE,
    )

    eligible_candidates = [
        result
        for result in candidate_results
        if (
            result["validation_task"]
            >= minimum_acceptable_task
        )
    ]

    # =====================================================
    # Select lambda
    # =====================================================
    #
    # Priority:
    #
    # 1. lowest identity leakage
    # 2. highest task performance
    # 3. smaller lambda

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

    standard_percentage = (
        standard_candidate[
            "retained_percentage"
        ]
    )

    # =====================================================
    # Print validation-selection table
    # =====================================================

    print(
        "\nValidation selection:"
    )

    print(
        f"Best validation task performance: "
        f"{100 * best_validation_task:.2f}%"
    )

    print(
        f"Minimum acceptable task performance: "
        f"{100 * minimum_acceptable_task:.2f}%"
    )

    print(
        "\nLambda | Val task | Val identity | "
        "Retained | Eligible | Selected"
    )

    print(
        "-------|----------|--------------|"
        "----------|----------|---------"
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
            f"{100 * result['validation_task']:>7.2f}% | "
            f"{100 * result['validation_identity']:>11.2f}% | "
            f"{result['retained_percentage']:>7.2f}% | "
            f"{str(eligible):>8} | "
            f"{str(selected):>8}"
        )

        validation_rows.append(
            {
                "outer_fold": (
                    test_fold_index + 1
                ),
                "lambda": (
                    result["lambda"]
                ),
                "validation_task_balanced_accuracy": (
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
                "training_time_seconds": (
                    result[
                        "time_seconds"
                    ]
                ),
            }
        )

    print(
        f"\nSelected lambda: "
        f"{selected_lambda}"
    )

    print(
        f"Selected identity-aware retention: "
        f"{selected_percentage:.2f}%"
    )

    print(
        f"Standard-SFD retention: "
        f"{standard_percentage:.2f}%"
    )

    # =====================================================
    # PHASE 2
    # Final Standard SFD model
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 2: REFIT FINAL MODELS"
    )

    print(
        "----------------------------------------"
    )

    print(
        "\nFitting final Standard SFD..."
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

    standard_model.fit(
        X_development,
        y_development,
    )

    # =====================================================
    # Final identity-aware model
    # =====================================================

    print(
        f"Fitting final Identity-aware SFD "
        f"with lambda={selected_lambda}..."
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

    # =====================================================
    # PHASE 3
    # Evaluate ONCE on untouched test subjects
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 3: FINAL OUTER-TEST EVALUATION"
    )

    print(
        "----------------------------------------"
    )

    # -----------------------------------------------------
    # Full unpruned ROCKET
    # -----------------------------------------------------

    full_predictions = (
        standard_model
        .full_classifier_
        .predict(
            standard_model
            ._prepare_X_full(
                X_test
            )
        )
    )

    full_task_score = (
        balanced_accuracy_score(
            y_test,
            full_predictions,
        )
    )

    # -----------------------------------------------------
    # Standard SFD
    # -----------------------------------------------------

    standard_predictions = (
        standard_model.predict(
            X_test
        )
    )

    standard_task_score = (
        balanced_accuracy_score(
            y_test,
            standard_predictions,
        )
    )

    # -----------------------------------------------------
    # Identity-aware SFD
    # -----------------------------------------------------

    identity_predictions = (
        identity_model.predict(
            X_test
        )
    )

    identity_task_score = (
        balanced_accuracy_score(
            y_test,
            identity_predictions,
        )
    )

    # =====================================================
    # Frozen test representations
    # =====================================================

    full_test_features = (
        standard_model
        ._prepare_X_full(
            X_test
        )
    )

    standard_test_features = (
        standard_model
        ._prepare_X(
            X_test
        )
    )

    identity_test_features = (
        identity_model
        ._prepare_X(
            X_test
        )
    )

    # =====================================================
    # Identity leakage
    # =====================================================

    full_identity_score = (
        identity_probe_accuracy(
            full_test_features,
            y_subject_test,
            random_state=RANDOM_SEED,
        )
    )

    standard_identity_score = (
        identity_probe_accuracy(
            standard_test_features,
            y_subject_test,
            random_state=RANDOM_SEED,
        )
    )

    identity_identity_score = (
        identity_probe_accuracy(
            identity_test_features,
            y_subject_test,
            random_state=RANDOM_SEED,
        )
    )

    # =====================================================
    # Actual retained percentages after refit
    # =====================================================

    standard_actual_retained = (
        100
        * standard_model
        .retained_ratios_[
            standard_model
            .selected_step_index_
        ]
    )

    identity_actual_retained = (
        100
        * identity_model
        .retained_ratios_[
            identity_model
            .selected_step_index_
        ]
    )

    # =====================================================
    # Print outer-fold result
    # =====================================================

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        f"Selected lambda: "
        f"{selected_lambda}"
    )

    print(
        f"Full ROCKET task balanced accuracy: "
        f"{100 * full_task_score:.2f}%"
    )

    print(
        f"Standard SFD task balanced accuracy: "
        f"{100 * standard_task_score:.2f}%"
    )

    print(
        f"Identity-aware SFD task balanced accuracy: "
        f"{100 * identity_task_score:.2f}%"
    )

    print(
        f"Full ROCKET identity probe: "
        f"{100 * full_identity_score:.2f}%"
    )

    print(
        f"Standard SFD identity probe: "
        f"{100 * standard_identity_score:.2f}%"
    )

    print(
        f"Identity-aware SFD identity probe: "
        f"{100 * identity_identity_score:.2f}%"
    )

    print(
        f"Standard SFD features retained: "
        f"{standard_actual_retained:.2f}%"
    )

    print(
        f"Identity-aware SFD features retained: "
        f"{identity_actual_retained:.2f}%"
    )

    print(
        "Identity chance level: 25.00%"
    )

    # =====================================================
    # Save fold result
    # =====================================================

    final_rows.append(
        {
            "outer_fold": (
                test_fold_index + 1
            ),
            "selected_lambda": (
                selected_lambda
            ),
            "standard_selected_percentage": (
                standard_percentage
            ),
            "identity_selected_percentage": (
                selected_percentage
            ),
            "standard_actual_retained_percentage": (
                standard_actual_retained
            ),
            "identity_actual_retained_percentage": (
                identity_actual_retained
            ),
            "full_task_balanced_accuracy": (
                full_task_score
            ),
            "standard_task_balanced_accuracy": (
                standard_task_score
            ),
            "identity_task_balanced_accuracy": (
                identity_task_score
            ),
            "full_identity_accuracy": (
                full_identity_score
            ),
            "standard_identity_accuracy": (
                standard_identity_score
            ),
            "identity_identity_accuracy": (
                identity_identity_score
            ),
        }
    )


# =========================================================
# Final cross-fold results
# =========================================================

full_task_scores = np.array(
    [
        row[
            "full_task_balanced_accuracy"
        ]
        for row in final_rows
    ]
)

standard_task_scores = np.array(
    [
        row[
            "standard_task_balanced_accuracy"
        ]
        for row in final_rows
    ]
)

identity_task_scores = np.array(
    [
        row[
            "identity_task_balanced_accuracy"
        ]
        for row in final_rows
    ]
)

full_identity_scores = np.array(
    [
        row[
            "full_identity_accuracy"
        ]
        for row in final_rows
    ]
)

standard_identity_scores = np.array(
    [
        row[
            "standard_identity_accuracy"
        ]
        for row in final_rows
    ]
)

identity_identity_scores = np.array(
    [
        row[
            "identity_identity_accuracy"
        ]
        for row in final_rows
    ]
)

standard_retained = np.array(
    [
        row[
            "standard_actual_retained_percentage"
        ]
        for row in final_rows
    ]
)

identity_retained = np.array(
    [
        row[
            "identity_actual_retained_percentage"
        ]
        for row in final_rows
    ]
)

selected_lambdas = [
    row[
        "selected_lambda"
    ]
    for row in final_rows
]


print(
    "\n\n"
    "============================================================"
)

print(
    "FINAL VALIDATION-SELECTED RESULTS"
)

print(
    "============================================================"
)

for row in final_rows:

    print(
        f"Fold {row['outer_fold']}: "
        f"lambda={row['selected_lambda']} | "
        f"Full task="
        f"{100 * row['full_task_balanced_accuracy']:.2f}% | "
        f"Standard task="
        f"{100 * row['standard_task_balanced_accuracy']:.2f}% | "
        f"Identity-aware task="
        f"{100 * row['identity_task_balanced_accuracy']:.2f}% | "
        f"Full identity="
        f"{100 * row['full_identity_accuracy']:.2f}% | "
        f"Standard identity="
        f"{100 * row['standard_identity_accuracy']:.2f}% | "
        f"Identity-aware identity="
        f"{100 * row['identity_identity_accuracy']:.2f}%"
    )


print(
    "\nSelected lambdas:",
    selected_lambdas,
)


print(
    "\nTASK BALANCED ACCURACY"
)

print(
    f"Full ROCKET: "
    f"{100 * np.mean(full_task_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(full_task_scores):.2f}%"
)

print(
    f"Standard SFD: "
    f"{100 * np.mean(standard_task_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(standard_task_scores):.2f}%"
)

print(
    f"Identity-aware SFD: "
    f"{100 * np.mean(identity_task_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(identity_task_scores):.2f}%"
)


print(
    "\nIDENTITY LEAKAGE"
)

print(
    f"Full ROCKET: "
    f"{100 * np.mean(full_identity_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(full_identity_scores):.2f}%"
)

print(
    f"Standard SFD: "
    f"{100 * np.mean(standard_identity_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(standard_identity_scores):.2f}%"
)

print(
    f"Identity-aware SFD: "
    f"{100 * np.mean(identity_identity_scores):.2f}% "
    f"+/- "
    f"{100 * np.std(identity_identity_scores):.2f}%"
)

print(
    "Identity chance level: 25.00%"
)


print(
    "\nFEATURES RETAINED"
)

print(
    f"Standard SFD: "
    f"{np.mean(standard_retained):.2f}% "
    f"+/- "
    f"{np.std(standard_retained):.2f}%"
)

print(
    f"Identity-aware SFD: "
    f"{np.mean(identity_retained):.2f}% "
    f"+/- "
    f"{np.std(identity_retained):.2f}%"
)


# =========================================================
# Save results
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


with open(
    VALIDATION_RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "outer_fold",
        "lambda",
        "validation_task_balanced_accuracy",
        "validation_identity_accuracy",
        "retained_percentage",
        "eligible",
        "selected",
        "training_time_seconds",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        validation_rows
    )


with open(
    FINAL_RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "outer_fold",
        "selected_lambda",
        "standard_selected_percentage",
        "identity_selected_percentage",
        "standard_actual_retained_percentage",
        "identity_actual_retained_percentage",
        "full_task_balanced_accuracy",
        "standard_task_balanced_accuracy",
        "identity_task_balanced_accuracy",
        "full_identity_accuracy",
        "standard_identity_accuracy",
        "identity_identity_accuracy",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        final_rows
    )


print(
    f"\nSaved validation results to: "
    f"{VALIDATION_RESULTS_CSV}"
)

print(
    f"Saved final results to: "
    f"{FINAL_RESULTS_CSV}"
)