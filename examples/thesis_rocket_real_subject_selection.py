"""
Subject-level model-selection analysis for real EEG.

IMPORTANT
---------
This script uses ONLY training and validation subjects.

For each outer-fold configuration and each lambda:

1. Generate the complete SFD pruning path.
2. Evaluate EVERY pruning step using subject-level validation
   balanced accuracy.
3. Select the pruning step using the existing Detach-ROCKET
   trade-off criterion, but with subject-level balanced accuracy
   instead of window-level accuracy.
4. Measure validation identity leakage at that selected step.
5. Select lambda:
      a. best subject-level validation task score
      b. retain lambdas within 5 percentage points
      c. lowest identity leakage
      d. tie: higher task score
      e. tie: smaller lambda

NO OUTER TEST SUBJECTS ARE EVALUATED.
"""

from pathlib import Path
import gc

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

OUTPUT_FILE = Path(
    "results/"
    "thesis_rocket_real_subject_level_selection.csv"
)

STEP_OUTPUT_FILE = Path(
    "results/"
    "thesis_rocket_real_subject_level_selection_steps.csv"
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
# Subject-level task metric
# ============================================================

def subject_level_balanced_accuracy(
    decision_scores,
    y_true,
    subject_ids,
):
    """
    Average decision scores across all windows belonging to
    each validation subject, producing one prediction per person.
    """

    true_labels = []
    predicted_labels = []

    for subject_id in np.unique(
        subject_ids
    ):

        mask = (
            subject_ids == subject_id
        )

        subject_labels = np.unique(
            y_true[mask]
        )

        if len(subject_labels) != 1:
            raise RuntimeError(
                f"{subject_id} has multiple task labels."
            )

        mean_score = float(
            decision_scores[
                mask
            ].mean()
        )

        true_label = int(
            subject_labels[0]
        )

        predicted_label = int(
            mean_score > 0
        )

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

    return float(
        balanced_accuracy_score(
            true_labels,
            predicted_labels,
        )
    )


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
# Calculate subject-level score at EVERY SFD step
# ============================================================

def calculate_subject_score_curve(
    X_train_features,
    y_train,
    X_val_features,
    y_val,
    val_subject_ids,
    importance_matrix,
    task_alpha,
):
    """
    For every SFD pruning step:

    - derive the retained feature mask
    - fit RidgeClassifier on TRAINING subjects only
    - obtain validation decision scores
    - aggregate predictions by subject
    - calculate subject-level balanced accuracy

    The Ridge alpha stays fixed to the alpha selected on the full
    training representation, matching the SFD path itself.
    """

    subject_scores = []

    for step_index in range(
        len(importance_matrix)
    ):

        feature_mask = (
            importance_matrix[
                step_index
            ]
            > 0
        )

        n_features = int(
            np.sum(
                feature_mask
            )
        )

        if n_features == 0:

            subject_scores.append(
                np.nan
            )

            continue

        classifier = RidgeClassifier(
            alpha=task_alpha
        )

        with quiet_ridge_warnings():

            classifier.fit(
                X_train_features[
                    :,
                    feature_mask,
                ],
                y_train,
            )

        decision_scores = (
            classifier.decision_function(
                X_val_features[
                    :,
                    feature_mask,
                ]
            )
        )

        score = (
            subject_level_balanced_accuracy(
                decision_scores,
                y_val,
                val_subject_ids,
            )
        )

        subject_scores.append(
            score
        )

    return np.asarray(
        subject_scores,
        dtype=float,
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
    len(
        np.unique(
            subject_ids
        )
    ),
)

print(
    "Lambda candidates:",
    IDENTITY_LAMBDAS,
)

print(
    "NO outer test subjects will be evaluated."
)


# ============================================================
# Results
# ============================================================

candidate_rows = []
step_rows = []


# ============================================================
# Outer-fold configurations
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
    # Training and validation subjects ONLY
    # --------------------------------------------------------

    val_subjects = (
        fold_table.loc[
            fold_table[
                "fold"
            ]
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

    y_subject_val = y_subject[
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

    # ========================================================
    # Compute common ROCKET representation
    # ========================================================

    print(
        "\nComputing ROCKET features..."
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

    task_alpha = float(
        standard_model
        .full_model_alpha_
    )

    print(
        "Task alpha:",
        task_alpha,
    )

    # --------------------------------------------------------
    # Identity alpha
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
        "Identity alpha:",
        identity_alpha,
    )

    fold_candidates = []

    # ========================================================
    # Lambda candidates
    # ========================================================

    for identity_lambda in (
        IDENTITY_LAMBDAS
    ):

        print(
            f"\nLambda = {identity_lambda}"
        )

        # ----------------------------------------------------
        # Obtain complete SFD path
        # ----------------------------------------------------

        if identity_lambda == 0.0:

            retained_ratios = (
                standard_model
                .retained_ratios_
            )

            importance_matrix = (
                standard_model
                .importance_matrix_
            )

            original_window_scores = (
                standard_model
                .val_scores_
            )

        else:

            task_classifier = RidgeClassifier(
                alpha=task_alpha
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
                original_window_scores,
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

        # ====================================================
        # NEW:
        # subject-level validation curve over ALL pruning steps
        # ====================================================

        print(
            "  Computing subject-level "
            "validation curve..."
        )

        subject_scores = (
            calculate_subject_score_curve(
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
                importance_matrix=(
                    importance_matrix
                ),
                task_alpha=task_alpha,
            )
        )

        # ----------------------------------------------------
        # Remove any impossible zero-feature steps
        # ----------------------------------------------------

        valid_steps = (
            ~np.isnan(
                subject_scores
            )
        )

        valid_indices = np.where(
            valid_steps
        )[0]

        valid_ratios = (
            retained_ratios[
                valid_steps
            ]
        )

        valid_subject_scores = (
            subject_scores[
                valid_steps
            ]
        )

        # ====================================================
        # Select pruning step using SUBJECT-LEVEL task metric
        # ====================================================

        (
            local_step_index,
            _,
        ) = select_optimal_pruning(
            valid_ratios,
            valid_subject_scores,
            trade_off=TRADE_OFF,
        )

        selected_step = int(
            valid_indices[
                local_step_index
            ]
        )

        selected_subject_task = float(
            subject_scores[
                selected_step
            ]
        )

        selected_window_task = float(
            original_window_scores[
                selected_step
            ]
        )

        selected_retention = float(
            100
            * retained_ratios[
                selected_step
            ]
        )

        selected_mask = (
            importance_matrix[
                selected_step
            ]
            > 0
        )

        selected_feature_count = int(
            np.sum(
                selected_mask
            )
        )

        # ====================================================
        # Identity leakage ONLY at selected pruning point
        # ====================================================

        selected_identity = (
            identity_probe_accuracy(
                X_val_features[
                    :,
                    selected_mask,
                ],
                y_subject_val,
                random_state=(
                    RANDOM_SEED
                ),
            )
        )

        print(
            "  Selected step:",
            selected_step,
        )

        print(
            "  Subject task:",
            f"{selected_subject_task * 100:.2f}%",
        )

        print(
            "  Window task:",
            f"{selected_window_task * 100:.2f}%",
        )

        print(
            "  Identity:",
            f"{selected_identity * 100:.2f}%",
        )

        print(
            "  Retained:",
            f"{selected_retention:.3f}%",
        )

        candidate = {
            "outer_fold": (
                outer_fold
            ),
            "lambda": (
                identity_lambda
            ),
            "selected_step": (
                selected_step
            ),
            "validation_subject_balanced_accuracy": (
                selected_subject_task
            ),
            "validation_window_accuracy": (
                selected_window_task
            ),
            "validation_identity_accuracy": (
                selected_identity
            ),
            "retained_percentage": (
                selected_retention
            ),
            "retained_feature_count": (
                selected_feature_count
            ),
        }

        candidate_rows.append(
            candidate
        )

        fold_candidates.append(
            candidate
        )

        # ----------------------------------------------------
        # Save complete subject-level pruning curve
        # ----------------------------------------------------

        for step_index in range(
            len(
                retained_ratios
            )
        ):

            step_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "lambda": (
                        identity_lambda
                    ),
                    "step": (
                        step_index
                    ),
                    "retained_percentage": (
                        100
                        * retained_ratios[
                            step_index
                        ]
                    ),
                    "window_validation_accuracy": (
                        original_window_scores[
                            step_index
                        ]
                    ),
                    "subject_validation_balanced_accuracy": (
                        subject_scores[
                            step_index
                        ]
                    ),
                    "subject_step_selected": (
                        step_index
                        == selected_step
                    ),
                }
            )

    # ========================================================
    # Lambda selection using SUBJECT task metric
    # ========================================================

    best_task = max(
        result[
            "validation_subject_balanced_accuracy"
        ]
        for result
        in fold_candidates
    )

    minimum_task = max(
        0.0,
        best_task
        - TASK_TOLERANCE,
    )

    eligible = [
        result
        for result
        in fold_candidates
        if (
            result[
                "validation_subject_balanced_accuracy"
            ]
            >= minimum_task
        )
    ]

    selected_candidate = sorted(
        eligible,
        key=lambda result: (
            result[
                "validation_identity_accuracy"
            ],
            -result[
                "validation_subject_balanced_accuracy"
            ],
            result[
                "lambda"
            ],
        ),
    )[0]

    # --------------------------------------------------------
    # Mark candidates
    # --------------------------------------------------------

    for result in fold_candidates:

        result[
            "eligible"
        ] = (
            result
            in eligible
        )

        result[
            "selected"
        ] = (
            result
            is selected_candidate
        )

    # ========================================================
    # Print fold table
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "SUBJECT-LEVEL SELECTION"
    )

    print(
        "----------------------------------------"
    )

    fold_df = pd.DataFrame(
        fold_candidates
    )

    print(
        fold_df[
            [
                "lambda",
                "validation_subject_balanced_accuracy",
                "validation_identity_accuracy",
                "retained_percentage",
                "eligible",
                "selected",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest subject task:",
        f"{best_task * 100:.2f}%",
    )

    print(
        "Minimum acceptable:",
        f"{minimum_task * 100:.2f}%",
    )

    print(
        "Selected lambda:",
        selected_candidate[
            "lambda"
        ],
    )

    print(
        "Selected retention:",
        f"{selected_candidate['retained_percentage']:.3f}%",
    )

    # --------------------------------------------------------
    # Memory cleanup
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
# Save results
# ============================================================

candidate_df = pd.DataFrame(
    candidate_rows
)

step_df = pd.DataFrame(
    step_rows
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

candidate_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

step_df.to_csv(
    STEP_OUTPUT_FILE,
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
    "FINAL SUBJECT-LEVEL MODEL SELECTION"
)

print(
    "============================================================"
)


for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    fold_df = candidate_df[
        candidate_df[
            "outer_fold"
        ]
        == outer_fold
    ]

    selected = fold_df[
        fold_df[
            "selected"
        ]
    ].iloc[0]

    # lambda=0 is also the subject-selected
    # Standard SFD candidate.
    standard = fold_df[
        np.isclose(
            fold_df[
                "lambda"
            ],
            0.0,
        )
    ].iloc[0]

    print(
        f"\nFold {outer_fold}:"
    )

    print(
        "  Standard SFD retention:",
        f"{standard['retained_percentage']:.3f}%",
    )

    print(
        "  Standard validation task:",
        f"{standard['validation_subject_balanced_accuracy'] * 100:.2f}%",
    )

    print(
        "  Selected identity lambda:",
        selected[
            "lambda"
        ],
    )

    print(
        "  Identity retention:",
        f"{selected['retained_percentage']:.3f}%",
    )

    print(
        "  Identity validation task:",
        f"{selected['validation_subject_balanced_accuracy'] * 100:.2f}%",
    )

    print(
        "  Identity validation leakage:",
        f"{selected['validation_identity_accuracy'] * 100:.2f}%",
    )


selected_lambdas = (
    candidate_df[
        candidate_df[
            "selected"
        ]
    ]
    .sort_values(
        "outer_fold"
    )[
        "lambda"
    ]
    .tolist()
)

standard_retentions = (
    candidate_df[
        np.isclose(
            candidate_df[
                "lambda"
            ],
            0.0,
        )
    ]
    .sort_values(
        "outer_fold"
    )[
        "retained_percentage"
    ]
    .tolist()
)

identity_retentions = (
    candidate_df[
        candidate_df[
            "selected"
        ]
    ]
    .sort_values(
        "outer_fold"
    )[
        "retained_percentage"
    ]
    .tolist()
)

print(
    "\nSelected lambdas:"
)

print(
    selected_lambdas
)

print(
    "\nSubject-selected Standard SFD retentions:"
)

print(
    standard_retentions
)

print(
    "\nSubject-selected Identity-aware retentions:"
)

print(
    identity_retentions
)

print(
    "\nSaved candidate results to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved complete pruning curves to:"
)

print(
    STEP_OUTPUT_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "No outer-test subjects were evaluated."
)