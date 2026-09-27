"""Final multi-seed synthetic experiment for identity-aware SFD.

This experiment evaluates:

    Full ROCKET
    Standard SFD
    Identity-aware SFD

under three controlled subject-identity strengths:

    weak
    medium
    strong

Five fresh synthetic random seeds are used. Seed 42 is intentionally excluded
because it was used during method development and pilot experiments.

For every synthetic seed and identity condition:

1. Five balanced subject-wise outer folds are constructed.
2. One fold is held out for final testing.
3. One fold is used for validation.
4. Three folds are used for training.
5. Lambda is selected using validation data only.
6. Final models are refit using all non-test subjects.
7. The held-out test subjects are evaluated once.

Lambda-selection rule
---------------------
1. Find the best validation task accuracy among all lambda candidates.
2. Keep candidates within 5 percentage points of that best task accuracy.
3. Among eligible candidates, select the one with the lowest validation
   subject-identity probe accuracy.
4. Break ties using:
      a. higher validation task accuracy
      b. smaller lambda

The lambda rule and candidate grid were fixed before running the fresh seeds.
"""

import csv
import os
from collections import Counter

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

# Fresh evaluation seeds.
# Seed 42 was used during development and is intentionally excluded.
SYNTHETIC_SEEDS = [
    43,
    44,
    45,
    46,
    47,
]

IDENTITY_CONDITIONS = [
    "weak",
    "medium",
    "strong",
]

TASK_AMPLITUDE = 0.50

N_KERNELS = 10_000
N_OUTER_FOLDS = 5

# Keep the ROCKET randomization fixed so that the main
# experimental variation comes from the synthetic datasets.
ROCKET_SEED = 42

IDENTITY_LAMBDAS = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
]

TASK_TOLERANCE = 0.05

RESULTS_DIR = "results"

VALIDATION_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_synthetic_validation.csv",
)

FOLD_RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_synthetic_folds.csv",
)

SEED_RESULTS_CSV = os.path.join(
    RESULTS_DIR,
    "thesis_rocket_synthetic_seed_summary.csv",
)


# =========================================================
# Helper functions
# =========================================================


def add_channel_dimension(X_data):
    """Convert (recordings, timepoints) to ROCKET input."""

    return X_data[:, np.newaxis, :]


def create_subject_folds(
    subject_classes,
    random_seed,
):
    """Create five balanced subject-wise folds."""

    rng = np.random.default_rng(
        random_seed
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

    return subject_folds


def identity_probe_accuracy(
    features,
    subject_labels,
    random_state=ROCKET_SEED,
):
    """Estimate recoverable subject identity from frozen features."""

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


def mean(values):
    """Return a float mean."""

    return float(
        np.mean(values)
    )


def sample_std(values):
    """Return sample standard deviation."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


# =========================================================
# Result storage
# =========================================================

validation_rows = []
fold_rows = []
seed_rows = []


# =========================================================
# Main experiment
# =========================================================

print(
    "\n============================================================"
)

print(
    "MULTI-SEED SYNTHETIC EXPERIMENT"
)

print(
    "============================================================"
)

print(
    "Synthetic seeds:",
    SYNTHETIC_SEEDS,
)

print(
    "Identity conditions:",
    IDENTITY_CONDITIONS,
)

print(
    "Lambda candidates:",
    IDENTITY_LAMBDAS,
)

print(
    f"Task tolerance: "
    f"{100 * TASK_TOLERANCE:.1f} percentage points"
)


# ---------------------------------------------------------
# Synthetic seeds
# ---------------------------------------------------------

for synthetic_seed in SYNTHETIC_SEEDS:

    print(
        "\n\n"
        "############################################################"
    )

    print(
        f"SYNTHETIC SEED {synthetic_seed}"
    )

    print(
        "############################################################"
    )

    (
        datasets,
        y_task,
        y_subject,
        subject_classes,
        identity_frequencies,
    ) = generate_synthetic_datasets(
        task_amplitude=TASK_AMPLITUDE,
        random_seed=synthetic_seed,
    )

    # The same subject folds are used for weak, medium,
    # and strong conditions within this seed.
    subject_folds = create_subject_folds(
        subject_classes,
        random_seed=synthetic_seed,
    )

    print(
        "\nSubject folds:"
    )

    for fold_index, subjects in enumerate(
        subject_folds
    ):
        print(
            f"  Fold {fold_index + 1}:",
            [
                f"S{s + 1:02d}"
                for s in subjects
            ],
        )

    # -----------------------------------------------------
    # Identity-strength conditions
    # -----------------------------------------------------

    for identity_condition in (
        IDENTITY_CONDITIONS
    ):

        print(
            "\n\n"
            "============================================================"
        )

        print(
            f"SEED {synthetic_seed} | "
            f"IDENTITY CONDITION: "
            f"{identity_condition.upper()}"
        )

        print(
            "============================================================"
        )

        X = datasets[
            identity_condition
        ]

        condition_fold_rows = []

        # =================================================
        # Outer test folds
        # =================================================

        for test_fold_index in range(
            N_OUTER_FOLDS
        ):

            print(
                "\n----------------------------------------"
            )

            print(
                f"Outer fold "
                f"{test_fold_index + 1}/"
                f"{N_OUTER_FOLDS}"
            )

            print(
                "----------------------------------------"
            )

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

            test_subjects = (
                subject_folds[
                    test_fold_index
                ]
            )

            val_subjects = (
                subject_folds[
                    val_fold_index
                ]
            )

            train_subjects = (
                np.concatenate(
                    [
                        subject_folds[i]
                        for i
                        in train_fold_indices
                    ]
                )
            )

            development_subjects = (
                np.concatenate(
                    [
                        train_subjects,
                        val_subjects,
                    ]
                )
            )

            # ---------------------------------------------
            # Masks
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            X_train = add_channel_dimension(
                X[train_mask]
            )

            y_train = y_task[
                train_mask
            ]

            y_subject_train = y_subject[
                train_mask
            ]

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            X_val = add_channel_dimension(
                X[val_mask]
            )

            y_val = y_task[
                val_mask
            ]

            y_subject_val = y_subject[
                val_mask
            ]

            # ---------------------------------------------
            # Development = train + validation
            # ---------------------------------------------

            X_development = (
                add_channel_dimension(
                    X[development_mask]
                )
            )

            y_development = y_task[
                development_mask
            ]

            y_subject_development = (
                y_subject[
                    development_mask
                ]
            )

            # ---------------------------------------------
            # Final test
            # ---------------------------------------------

            X_test = add_channel_dimension(
                X[test_mask]
            )

            y_test = y_task[
                test_mask
            ]

            y_subject_test = y_subject[
                test_mask
            ]

            # =================================================
            # PHASE 1
            # Validation-only lambda selection
            # =================================================

            candidate_results = []

            for identity_lambda in (
                IDENTITY_LAMBDAS
            ):

                rocket = Rocket(
                    n_kernels=N_KERNELS,
                    random_state=ROCKET_SEED,
                )

                candidate_model = (
                    DetachRocket(
                        transformer=rocket,
                        trade_off=0.1,
                        verbose=False,
                    )
                )

                candidate_model.fit(
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                    y_identity=(
                        y_subject_train
                    ),
                    identity_lambda=(
                        identity_lambda
                    ),
                )

                selected_step = (
                    candidate_model
                    .selected_step_index_
                )

                # Validation folds are exactly balanced
                # by task class, so validation accuracy
                # equals balanced accuracy.
                validation_task = float(
                    candidate_model
                    .val_scores_[
                        selected_step
                    ]
                )

                validation_features = (
                    candidate_model
                    ._prepare_X(
                        X_val
                    )
                )

                validation_identity = (
                    identity_probe_accuracy(
                        validation_features,
                        y_subject_val,
                    )
                )

                retained_percentage = (
                    100
                    * candidate_model
                    .retained_ratios_[
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

            # =================================================
            # Lambda eligibility
            # =================================================

            standard_candidate = next(
                result
                for result
                in candidate_results
                if result["lambda"] == 0.0
            )

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

            standard_percentage = (
                standard_candidate[
                    "retained_percentage"
                ]
            )

            # ---------------------------------------------
            # Save candidate validation results
            # ---------------------------------------------

            for result in candidate_results:

                eligible = (
                    result
                    in eligible_candidates
                )

                selected = (
                    result
                    is selected_candidate
                )

                validation_rows.append(
                    {
                        "synthetic_seed": (
                            synthetic_seed
                        ),
                        "identity_condition": (
                            identity_condition
                        ),
                        "outer_fold": (
                            test_fold_index + 1
                        ),
                        "lambda": (
                            result[
                                "lambda"
                            ]
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
                    }
                )

            print(
                f"Selected lambda: "
                f"{selected_lambda}"
            )

            print(
                f"Validation best task: "
                f"{100 * best_validation_task:.2f}%"
            )

            print(
                f"Validation identity of selected model: "
                f"{100 * selected_candidate['validation_identity']:.2f}%"
            )

            # =================================================
            # PHASE 2
            # Refit final Standard SFD
            # =================================================

            standard_rocket = Rocket(
                n_kernels=N_KERNELS,
                random_state=ROCKET_SEED,
            )

            standard_model = DetachRocket(
                transformer=standard_rocket,
                set_percentage=(
                    standard_percentage
                ),
                verbose=False,
            )

            standard_model.fit(
                X_development,
                y_development,
            )

            # =================================================
            # Refit final identity-aware SFD
            # =================================================

            identity_rocket = Rocket(
                n_kernels=N_KERNELS,
                random_state=ROCKET_SEED,
            )

            identity_model = DetachRocket(
                transformer=identity_rocket,
                set_percentage=(
                    selected_percentage
                ),
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

            # =================================================
            # PHASE 3
            # Final untouched test evaluation
            # =================================================

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

            standard_predictions = (
                standard_model.predict(
                    X_test
                )
            )

            identity_predictions = (
                identity_model.predict(
                    X_test
                )
            )

            full_task_score = (
                balanced_accuracy_score(
                    y_test,
                    full_predictions,
                )
            )

            standard_task_score = (
                balanced_accuracy_score(
                    y_test,
                    standard_predictions,
                )
            )

            identity_task_score = (
                balanced_accuracy_score(
                    y_test,
                    identity_predictions,
                )
            )

            # ---------------------------------------------
            # Frozen representations
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Identity leakage
            # ---------------------------------------------

            full_identity_score = (
                identity_probe_accuracy(
                    full_test_features,
                    y_subject_test,
                )
            )

            standard_identity_score = (
                identity_probe_accuracy(
                    standard_test_features,
                    y_subject_test,
                )
            )

            identity_identity_score = (
                identity_probe_accuracy(
                    identity_test_features,
                    y_subject_test,
                )
            )

            # ---------------------------------------------
            # Actual retained percentages
            # ---------------------------------------------

            standard_retained = (
                100
                * standard_model
                .retained_ratios_[
                    standard_model
                    .selected_step_index_
                ]
            )

            identity_retained = (
                100
                * identity_model
                .retained_ratios_[
                    identity_model
                    .selected_step_index_
                ]
            )

            row = {
                "synthetic_seed": (
                    synthetic_seed
                ),
                "identity_condition": (
                    identity_condition
                ),
                "outer_fold": (
                    test_fold_index + 1
                ),
                "selected_lambda": (
                    selected_lambda
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
                "standard_retained_percentage": (
                    standard_retained
                ),
                "identity_retained_percentage": (
                    identity_retained
                ),
            }

            fold_rows.append(
                row
            )

            condition_fold_rows.append(
                row
            )

            print(
                f"Test task: "
                f"Full={100 * full_task_score:.2f}% | "
                f"Standard={100 * standard_task_score:.2f}% | "
                f"Identity-aware={100 * identity_task_score:.2f}%"
            )

            print(
                f"Test identity: "
                f"Full={100 * full_identity_score:.2f}% | "
                f"Standard={100 * standard_identity_score:.2f}% | "
                f"Identity-aware={100 * identity_identity_score:.2f}%"
            )

        # =================================================
        # Seed-level summary for this condition
        # =================================================

        seed_row = {
            "synthetic_seed": (
                synthetic_seed
            ),
            "identity_condition": (
                identity_condition
            ),
            "full_task_mean": mean(
                [
                    row[
                        "full_task_balanced_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "standard_task_mean": mean(
                [
                    row[
                        "standard_task_balanced_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "identity_task_mean": mean(
                [
                    row[
                        "identity_task_balanced_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "full_identity_mean": mean(
                [
                    row[
                        "full_identity_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "standard_identity_mean": mean(
                [
                    row[
                        "standard_identity_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "identity_identity_mean": mean(
                [
                    row[
                        "identity_identity_accuracy"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "standard_retained_mean": mean(
                [
                    row[
                        "standard_retained_percentage"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
            "identity_retained_mean": mean(
                [
                    row[
                        "identity_retained_percentage"
                    ]
                    for row
                    in condition_fold_rows
                ]
            ),
        }

        seed_rows.append(
            seed_row
        )

        print(
            "\nSeed-condition summary:"
        )

        print(
            f"  Task: "
            f"Full={100 * seed_row['full_task_mean']:.2f}% | "
            f"Standard={100 * seed_row['standard_task_mean']:.2f}% | "
            f"Identity-aware={100 * seed_row['identity_task_mean']:.2f}%"
        )

        print(
            f"  Identity: "
            f"Full={100 * seed_row['full_identity_mean']:.2f}% | "
            f"Standard={100 * seed_row['standard_identity_mean']:.2f}% | "
            f"Identity-aware={100 * seed_row['identity_identity_mean']:.2f}%"
        )


# =========================================================
# Final summary across fresh synthetic seeds
# =========================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "FINAL MULTI-SEED SYNTHETIC SUMMARY"
)

print(
    "============================================================"
)

print(
    "\nMean +/- sample SD across the five synthetic-seed means."
)


for identity_condition in (
    IDENTITY_CONDITIONS
):

    condition_seed_rows = [
        row
        for row
        in seed_rows
        if (
            row[
                "identity_condition"
            ]
            == identity_condition
        )
    ]

    full_task = [
        row["full_task_mean"]
        for row
        in condition_seed_rows
    ]

    standard_task = [
        row["standard_task_mean"]
        for row
        in condition_seed_rows
    ]

    identity_task = [
        row["identity_task_mean"]
        for row
        in condition_seed_rows
    ]

    full_identity = [
        row["full_identity_mean"]
        for row
        in condition_seed_rows
    ]

    standard_identity = [
        row[
            "standard_identity_mean"
        ]
        for row
        in condition_seed_rows
    ]

    identity_identity = [
        row[
            "identity_identity_mean"
        ]
        for row
        in condition_seed_rows
    ]

    standard_retained = [
        row[
            "standard_retained_mean"
        ]
        for row
        in condition_seed_rows
    ]

    identity_retained = [
        row[
            "identity_retained_mean"
        ]
        for row
        in condition_seed_rows
    ]

    condition_lambdas = [
        row[
            "selected_lambda"
        ]
        for row
        in fold_rows
        if (
            row[
                "identity_condition"
            ]
            == identity_condition
        )
    ]

    lambda_counts = Counter(
        condition_lambdas
    )

    print(
        "\n------------------------------------------------------------"
    )

    print(
        identity_condition.upper()
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "Task balanced accuracy:"
    )

    print(
        f"  Full ROCKET: "
        f"{100 * mean(full_task):.2f}% "
        f"+/- "
        f"{100 * sample_std(full_task):.2f}%"
    )

    print(
        f"  Standard SFD: "
        f"{100 * mean(standard_task):.2f}% "
        f"+/- "
        f"{100 * sample_std(standard_task):.2f}%"
    )

    print(
        f"  Identity-aware SFD: "
        f"{100 * mean(identity_task):.2f}% "
        f"+/- "
        f"{100 * sample_std(identity_task):.2f}%"
    )

    print(
        "Identity leakage:"
    )

    print(
        f"  Full ROCKET: "
        f"{100 * mean(full_identity):.2f}% "
        f"+/- "
        f"{100 * sample_std(full_identity):.2f}%"
    )

    print(
        f"  Standard SFD: "
        f"{100 * mean(standard_identity):.2f}% "
        f"+/- "
        f"{100 * sample_std(standard_identity):.2f}%"
    )

    print(
        f"  Identity-aware SFD: "
        f"{100 * mean(identity_identity):.2f}% "
        f"+/- "
        f"{100 * sample_std(identity_identity):.2f}%"
    )

    print(
        "Features retained:"
    )

    print(
        f"  Standard SFD: "
        f"{mean(standard_retained):.2f}% "
        f"+/- "
        f"{sample_std(standard_retained):.2f}%"
    )

    print(
        f"  Identity-aware SFD: "
        f"{mean(identity_retained):.2f}% "
        f"+/- "
        f"{sample_std(identity_retained):.2f}%"
    )

    print(
        "Selected lambda counts:",
        dict(
            sorted(
                lambda_counts.items()
            )
        ),
    )


# =========================================================
# Save results
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


with open(
    VALIDATION_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "synthetic_seed",
        "identity_condition",
        "outer_fold",
        "lambda",
        "validation_task_balanced_accuracy",
        "validation_identity_accuracy",
        "retained_percentage",
        "eligible",
        "selected",
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
    FOLD_RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "synthetic_seed",
        "identity_condition",
        "outer_fold",
        "selected_lambda",
        "full_task_balanced_accuracy",
        "standard_task_balanced_accuracy",
        "identity_task_balanced_accuracy",
        "full_identity_accuracy",
        "standard_identity_accuracy",
        "identity_identity_accuracy",
        "standard_retained_percentage",
        "identity_retained_percentage",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        fold_rows
    )


with open(
    SEED_RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = [
        "synthetic_seed",
        "identity_condition",
        "full_task_mean",
        "standard_task_mean",
        "identity_task_mean",
        "full_identity_mean",
        "standard_identity_mean",
        "identity_identity_mean",
        "standard_retained_mean",
        "identity_retained_mean",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        seed_rows
    )


print(
    "\nSaved validation results to:",
    VALIDATION_CSV,
)

print(
    "Saved fold-level results to:",
    FOLD_RESULTS_CSV,
)

print(
    "Saved seed-level summaries to:",
    SEED_RESULTS_CSV,
)