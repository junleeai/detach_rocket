"""Pilot ROCKET + identity-aware SFD lambda sweep for the thesis.

This script is a DEVELOPMENT / SENSITIVITY experiment.
It intentionally evaluates several identity-penalty values on the outer test
folds so that we can inspect whether the implementation behaves sensibly.

Do not use this sweep to select the final thesis lambda. The final experiment
must choose lambda using training/validation subjects only and evaluate the
chosen value once on unseen outer-test subjects.
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

# Pilot sensitivity grid only.
# lambda = 0 is the standard-SFD reference because the identity-aware
# ranking reduces to ordinary task-only SFD when the penalty is zero.
IDENTITY_LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]

RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_lambda_sweep.csv",
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

# Use the strong subject-identity condition.
X = datasets[IDENTITY_CONDITION]

print("\nSynthetic experiment settings:")
print(f"  Task amplitude: {TASK_AMPLITUDE}")
print(f"  Identity condition: {IDENTITY_CONDITION}")
print(f"  ROCKET kernels: {N_KERNELS}")
print(f"  Lambda grid: {IDENTITY_LAMBDAS}")

print(
    "\nIMPORTANT: This is a PILOT lambda sensitivity sweep. "
    "Do not use the outer-test results below to choose the final thesis lambda."
)


# =========================================================
# Create balanced subject-wise folds
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

class_0_subjects = np.where(
    subject_classes == 0
)[0]

class_1_subjects = np.where(
    subject_classes == 1
)[0]

# Shuffle subjects within each class.
rng.shuffle(class_0_subjects)
rng.shuffle(class_1_subjects)

subject_folds = []

for fold_index in range(N_OUTER_FOLDS):
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

    subject_folds.append(fold)


# ---------------------------------------------------------
# Show the folds
# ---------------------------------------------------------

print("\nSubject-wise folds:\n")

for i, fold in enumerate(subject_folds):
    print(
        f"Fold {i + 1}:",
        [f"S{s + 1:02d}" for s in fold],
    )

    print(
        "  Classes:",
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
    """Estimate subject-identity leakage from a frozen representation.

    A multinomial logistic-regression probe is trained/evaluated with
    five-fold stratified cross-validation within the four held-out subjects.
    Because all four subjects contribute the same number of recordings,
    chance identity accuracy is 25%.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    scores = []

    for probe_train_idx, probe_test_idx in cv.split(
        features,
        subject_labels,
    ):
        X_probe_train = features[probe_train_idx]
        X_probe_test = features[probe_test_idx]

        y_probe_train = subject_labels[probe_train_idx]
        y_probe_test = subject_labels[probe_test_idx]

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

        scores.append(score)

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
    )


# =========================================================
# Run one lambda across all outer folds
# =========================================================


def run_lambda_experiment(identity_lambda):
    """Run all five outer folds for one identity-penalty value."""

    full_scores = []
    sfd_scores = []
    retained_percentages = []

    full_identity_scores = []
    sfd_identity_scores = []

    fold_rows = []

    method_name = (
        "Standard SFD"
        if identity_lambda == 0.0
        else "Identity-aware SFD"
    )

    print(
        "\n\n############################################################"
    )
    print(
        f"LAMBDA = {identity_lambda} | {method_name}"
    )
    print(
        "############################################################"
    )

    for test_fold_index in range(N_OUTER_FOLDS):
        # -----------------------------------------------------
        # Define train / validation / test folds
        # -----------------------------------------------------

        # The next fold is used for validation.
        val_fold_index = (
            test_fold_index + 1
        ) % N_OUTER_FOLDS

        # The other three folds are used for training.
        train_fold_indices = [
            i
            for i in range(N_OUTER_FOLDS)
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

        # -----------------------------------------------------
        # Create recording-level masks
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

        # -----------------------------------------------------
        # Split the time-series recordings
        # -----------------------------------------------------

        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]

        y_train = y_task[train_mask]
        y_val = y_task[val_mask]
        y_test = y_task[test_mask]

        # Training subject labels are used by identity-aware SFD.
        y_subject_train = y_subject[train_mask]

        # Test subject labels are used only by the post-hoc leakage probe.
        y_subject_test = y_subject[test_mask]

        # -----------------------------------------------------
        # Add channel dimension
        # -----------------------------------------------------

        # Current shape:
        # (recordings, timepoints)
        #
        # Required ROCKET shape:
        # (recordings, channels, timepoints)

        X_train = X_train[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

        # -----------------------------------------------------
        # Print fold information
        # -----------------------------------------------------

        print(
            "\n========================================"
        )
        print(
            f"Lambda {identity_lambda} | "
            f"test fold {test_fold_index + 1}"
        )
        print(
            "========================================"
        )

        print(
            "Training subjects:",
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
            "Test subjects:",
            [
                f"S{s + 1:02d}"
                for s in test_subjects
            ],
        )

        print(
            "Training shape:",
            X_train.shape,
        )

        print(
            "Validation shape:",
            X_val.shape,
        )

        print(
            "Test shape:",
            X_test.shape,
        )

        # -----------------------------------------------------
        # Initialize ROCKET
        # -----------------------------------------------------

        rocket = Rocket(
            n_kernels=N_KERNELS,
            random_state=RANDOM_SEED,
        )

        # -----------------------------------------------------
        # Initialize Detach-ROCKET
        # -----------------------------------------------------

        detach_model = DetachRocket(
            transformer=rocket,
            trade_off=0.1,
            verbose=False,
        )

        # -----------------------------------------------------
        # Train ROCKET + SFD
        # -----------------------------------------------------

        print(
            f"\nTraining ROCKET + {method_name} "
            f"(lambda={identity_lambda})..."
        )

        start_time = time.time()

        detach_model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            y_identity=y_subject_train,
            identity_lambda=identity_lambda,
        )

        training_time = (
            time.time() - start_time
        )

        # =====================================================
        # TASK PERFORMANCE
        # =====================================================

        # Use explicit balanced accuracy for thesis reporting.
        # With the present balanced synthetic folds this is numerically
        # equivalent to ordinary accuracy, but it is the intended thesis
        # metric and remains valid if class balance changes later.

        full_predictions = (
            detach_model.full_classifier_.predict(
                detach_model._prepare_X_full(
                    X_test
                )
            )
        )

        sfd_predictions = detach_model.predict(
            X_test
        )

        full_score = balanced_accuracy_score(
            y_test,
            full_predictions,
        )

        sfd_score = balanced_accuracy_score(
            y_test,
            sfd_predictions,
        )

        # Percentage of ROCKET features retained by SFD.
        retained_percentage = (
            100
            * detach_model.retained_ratios_[
                detach_model.selected_step_index_
            ]
        )

        full_scores.append(
            full_score
        )

        sfd_scores.append(
            sfd_score
        )

        retained_percentages.append(
            retained_percentage
        )

        # =====================================================
        # IDENTITY-LEAKAGE EVALUATION
        # =====================================================

        # Full ROCKET representation.
        full_test_features = (
            detach_model._prepare_X_full(
                X_test
            )
        )

        # SFD-pruned representation.
        sfd_test_features = (
            detach_model._prepare_X(
                X_test
            )
        )

        (
            full_identity_score,
            full_identity_std,
        ) = identity_probe_accuracy(
            full_test_features,
            y_subject_test,
            random_state=RANDOM_SEED,
        )

        (
            sfd_identity_score,
            sfd_identity_std,
        ) = identity_probe_accuracy(
            sfd_test_features,
            y_subject_test,
            random_state=RANDOM_SEED,
        )

        full_identity_scores.append(
            full_identity_score
        )

        sfd_identity_scores.append(
            sfd_identity_score
        )

        # =====================================================
        # PRINT FOLD RESULTS
        # =====================================================

        print("\nFold results:")

        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )

        print(
            f"Full ROCKET task balanced accuracy: "
            f"{100 * full_score:.2f}%"
        )

        print(
            f"{method_name} task balanced accuracy: "
            f"{100 * sfd_score:.2f}%"
        )

        print(
            f"Features retained by SFD: "
            f"{retained_percentage:.2f}%"
        )

        print(
            f"Full ROCKET identity probe: "
            f"{100 * full_identity_score:.2f}% "
            f"+/- "
            f"{100 * full_identity_std:.2f}%"
        )

        print(
            f"{method_name} identity probe: "
            f"{100 * sfd_identity_score:.2f}% "
            f"+/- "
            f"{100 * sfd_identity_std:.2f}%"
        )

        print(
            "Identity chance level: 25.00%"
        )

        fold_rows.append(
            {
                "lambda": identity_lambda,
                "fold": test_fold_index + 1,
                "method": method_name,
                "full_task_balanced_accuracy": full_score,
                "sfd_task_balanced_accuracy": sfd_score,
                "retained_percentage": retained_percentage,
                "full_identity_accuracy": full_identity_score,
                "sfd_identity_accuracy": sfd_identity_score,
                "training_time_seconds": training_time,
            }
        )

    # =========================================================
    # Lambda-level summary
    # =========================================================

    print(
        "\n========================================"
    )
    print(
        f"5-FOLD RESULTS | LAMBDA = {identity_lambda}"
    )
    print(
        "========================================"
    )

    for i in range(N_OUTER_FOLDS):
        print(
            f"Fold {i + 1}: "
            f"Full={100 * full_scores[i]:.2f}% | "
            f"{method_name}={100 * sfd_scores[i]:.2f}% | "
            f"Retained={retained_percentages[i]:.2f}% | "
            f"Full identity={100 * full_identity_scores[i]:.2f}% | "
            f"{method_name} identity="
            f"{100 * sfd_identity_scores[i]:.2f}%"
        )

    print("\nTask-performance results:")

    print(
        f"Full ROCKET balanced accuracy: "
        f"{100 * np.mean(full_scores):.2f}% "
        f"+/- "
        f"{100 * np.std(full_scores):.2f}%"
    )

    print(
        f"{method_name} balanced accuracy: "
        f"{100 * np.mean(sfd_scores):.2f}% "
        f"+/- "
        f"{100 * np.std(sfd_scores):.2f}%"
    )

    print(
        f"Features retained: "
        f"{np.mean(retained_percentages):.2f}% "
        f"+/- "
        f"{np.std(retained_percentages):.2f}%"
    )

    print("\nIdentity-leakage results:")

    print(
        f"Full ROCKET identity probe: "
        f"{100 * np.mean(full_identity_scores):.2f}% "
        f"+/- "
        f"{100 * np.std(full_identity_scores):.2f}%"
    )

    print(
        f"{method_name} identity probe: "
        f"{100 * np.mean(sfd_identity_scores):.2f}% "
        f"+/- "
        f"{100 * np.std(sfd_identity_scores):.2f}%"
    )

    print(
        "Identity chance level: 25.00%"
    )

    summary = {
        "lambda": identity_lambda,
        "method": method_name,
        "full_task_mean": float(np.mean(full_scores)),
        "full_task_std": float(np.std(full_scores)),
        "sfd_task_mean": float(np.mean(sfd_scores)),
        "sfd_task_std": float(np.std(sfd_scores)),
        "retained_mean": float(np.mean(retained_percentages)),
        "retained_std": float(np.std(retained_percentages)),
        "full_identity_mean": float(np.mean(full_identity_scores)),
        "full_identity_std": float(np.std(full_identity_scores)),
        "sfd_identity_mean": float(np.mean(sfd_identity_scores)),
        "sfd_identity_std": float(np.std(sfd_identity_scores)),
    }

    return summary, fold_rows


# =========================================================
# Run the complete pilot lambda sweep
# =========================================================

all_summaries = []
all_fold_rows = []

for identity_lambda in IDENTITY_LAMBDAS:
    summary, fold_rows = run_lambda_experiment(
        identity_lambda
    )

    all_summaries.append(summary)
    all_fold_rows.extend(fold_rows)


# =========================================================
# Cross-lambda summary
# =========================================================

print(
    "\n\n============================================================"
)
print(
    "PILOT LAMBDA SENSITIVITY SUMMARY"
)
print(
    "============================================================"
)
print(
    "These outer-test results are for implementation/sensitivity "
    "analysis only. Do not select the final thesis lambda from this table."
)

print(
    "\n"
    "Lambda | Method             | Task bal. acc.      | "
    "Identity probe      | Retained"
)
print(
    "-------|--------------------|---------------------|"
    "---------------------|----------------"
)

for summary in all_summaries:
    print(
        f"{summary['lambda']:>6.2f} | "
        f"{summary['method']:<18} | "
        f"{100 * summary['sfd_task_mean']:>6.2f}% "
        f"+/- {100 * summary['sfd_task_std']:>5.2f}% | "
        f"{100 * summary['sfd_identity_mean']:>6.2f}% "
        f"+/- {100 * summary['sfd_identity_std']:>5.2f}% | "
        f"{summary['retained_mean']:>6.2f}% "
        f"+/- {summary['retained_std']:>5.2f}%"
    )


# =========================================================
# Save per-fold pilot results
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    fieldnames = [
        "lambda",
        "fold",
        "method",
        "full_task_balanced_accuracy",
        "sfd_task_balanced_accuracy",
        "retained_percentage",
        "full_identity_accuracy",
        "sfd_identity_accuracy",
        "training_time_seconds",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(all_fold_rows)

print(
    f"\nSaved per-fold pilot results to: {RESULTS_CSV}"
)