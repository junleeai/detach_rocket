from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    RidgeClassifierCV,
    LogisticRegression,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.convolution_based import Rocket


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
    "results/thesis_rocket_real_baseline.csv"
)

PREDICTION_FILE = Path(
    "results/thesis_rocket_real_baseline_subject_predictions.csv"
)

N_KERNELS = 10_000
RANDOM_SEED = 42
N_FOLDS = 5

ALPHAS = np.logspace(
    -10,
    10,
    20,
)


# ============================================================
# Helpers
# ============================================================

def evaluate_subject_level(
    decision_scores,
    y_true,
    subject_ids,
):
    """
    Convert window-level Ridge decision scores into one prediction
    per subject by averaging all 30 window scores.
    """

    rows = []

    for subject_id in np.unique(subject_ids):

        mask = subject_ids == subject_id

        labels = np.unique(
            y_true[mask]
        )

        if len(labels) != 1:
            raise RuntimeError(
                f"{subject_id} has multiple task labels."
            )

        mean_score = float(
            decision_scores[mask].mean()
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

    subject_df = pd.DataFrame(rows)

    score = balanced_accuracy_score(
        subject_df["true_label"],
        subject_df["predicted_label"],
    )

    return score, subject_df


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

print("Dataset shape:", X.shape)
print(
    "Subjects:",
    len(np.unique(subject_ids)),
)


# ============================================================
# Run five outer folds
# ============================================================

fold_results = []
all_subject_predictions = []

for outer_fold in range(
    1,
    N_FOLDS + 1,
):

    print(
        "\n"
        "========================================"
    )

    print(
        f"OUTER FOLD {outer_fold}"
    )

    print(
        "========================================"
    )

    test_fold = outer_fold

    val_fold = (
        outer_fold % N_FOLDS
    ) + 1

    # --------------------------------------------------------
    # Subject sets
    # --------------------------------------------------------

    test_subjects = fold_table.loc[
        fold_table["fold"] == test_fold,
        "subject",
    ].to_numpy()

    val_subjects = fold_table.loc[
        fold_table["fold"] == val_fold,
        "subject",
    ].to_numpy()

    train_subjects = fold_table.loc[
        ~fold_table["fold"].isin(
            [test_fold, val_fold]
        ),
        "subject",
    ].to_numpy()

    # Final Full ROCKET baseline uses all non-test subjects.
    development_subjects = np.concatenate(
        [
            train_subjects,
            val_subjects,
        ]
    )

    # --------------------------------------------------------
    # Verify no subject leakage
    # --------------------------------------------------------

    overlap = (
        set(development_subjects)
        & set(test_subjects)
    )

    if overlap:
        raise RuntimeError(
            f"Subject leakage: {overlap}"
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

    dev_mask = np.isin(
        subject_ids,
        development_subjects,
    )

    test_mask = np.isin(
        subject_ids,
        test_subjects,
    )

    X_dev = X[dev_mask]
    X_test = X[test_mask]

    y_dev = y_task[dev_mask]
    y_test = y_task[test_mask]

    test_subject_ids = subject_ids[
        test_mask
    ]

    print(
        "Development windows:",
        len(X_dev),
    )

    print(
        "Test windows:",
        len(X_test),
    )

    # --------------------------------------------------------
    # Full ROCKET representation
    # --------------------------------------------------------

    print(
        f"\nFitting ROCKET "
        f"({N_KERNELS:,} kernels)..."
    )

    rocket = Rocket(
        n_kernels=N_KERNELS,
        random_state=RANDOM_SEED,
    )

    X_dev_rocket = rocket.fit_transform(
        X_dev
    )

    X_test_rocket = rocket.transform(
        X_test
    )

    print(
        "Development ROCKET shape:",
        X_dev_rocket.shape,
    )

    print(
        "Test ROCKET shape:",
        X_test_rocket.shape,
    )

    if np.isnan(X_dev_rocket).any():
        raise RuntimeError(
            "NaNs found in development ROCKET features."
        )

    if np.isnan(X_test_rocket).any():
        raise RuntimeError(
            "NaNs found in test ROCKET features."
        )

    # --------------------------------------------------------
    # Scale using development data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev_rocket
    )

    X_test_scaled = scaler.transform(
        X_test_rocket
    )

    # --------------------------------------------------------
    # Task classifier
    # --------------------------------------------------------

    print(
        "\nFitting RidgeClassifierCV..."
    )

    classifier = RidgeClassifierCV(
        alphas=ALPHAS
    )

    classifier.fit(
        X_dev_scaled,
        y_dev,
    )

    print(
        "Selected alpha:",
        classifier.alpha_,
    )

    # --------------------------------------------------------
    # Window-level task performance
    # --------------------------------------------------------

    window_predictions = classifier.predict(
        X_test_scaled
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
        classifier.decision_function(
            X_test_scaled
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
    # Subject-identity leakage probe
    # --------------------------------------------------------

    print(
        "\nEvaluating subject-identity leakage..."
    )

    identity_labels = y_subject[
        test_mask
    ]

    n_test_identities = len(
        np.unique(identity_labels)
    )

    # Every outer test fold should contain 13 subjects.
    if n_test_identities != len(test_subjects):
        raise RuntimeError(
            "Identity-label count does not match "
            "the number of test subjects."
        )

    # Every test subject should have exactly 30 windows.
    identity_counts = pd.Series(
        identity_labels
    ).value_counts()

    if not (
        identity_counts == 30
    ).all():
        raise RuntimeError(
            "Not every test identity has 30 windows."
        )

    identity_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    identity_true = []
    identity_pred = []

    for (
        probe_train_index,
        probe_test_index,
    ) in identity_cv.split(
        X_test_scaled,
        identity_labels,
    ):

        probe = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        )

        probe.fit(
            X_test_scaled[
                probe_train_index
            ],
            identity_labels[
                probe_train_index
            ],
        )

        predictions_identity = probe.predict(
            X_test_scaled[
                probe_test_index
            ]
        )

        identity_true.extend(
            identity_labels[
                probe_test_index
            ]
        )

        identity_pred.extend(
            predictions_identity
        )

    identity_accuracy = accuracy_score(
        identity_true,
        identity_pred,
    )

    identity_chance = (
        1.0
        / n_test_identities
    )

    print(
        "Identity-probe accuracy:",
        f"{identity_accuracy * 100:.2f}%",
    )

    print(
        "Identity chance level:",
        f"{identity_chance * 100:.2f}%",
    )

    # --------------------------------------------------------
    # Store result
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
            "selected_alpha": (
                classifier.alpha_
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
        }
    )

    print(
        "\nWindow-level balanced accuracy:",
        f"{window_balanced_accuracy * 100:.2f}%",
    )

    print(
        "Subject-level balanced accuracy:",
        f"{subject_balanced_accuracy * 100:.2f}%",
    )

    # --------------------------------------------------------
    # Free large arrays before next fold
    # --------------------------------------------------------

    del rocket
    del scaler
    del classifier
    del probe

    del X_dev
    del X_test

    del X_dev_rocket
    del X_test_rocket

    del X_dev_scaled
    del X_test_scaled

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

# Every subject should appear exactly once as test data.
subject_test_counts = (
    predictions["subject"]
    .value_counts()
)

if not (
    subject_test_counts == 1
).all():
    raise RuntimeError(
        "Some subjects were tested more or less than once."
    )

if len(predictions) != 65:
    raise RuntimeError(
        f"Expected 65 test-subject predictions, "
        f"found {len(predictions)}."
    )


# ============================================================
# Pooled out-of-fold subject task accuracy
# ============================================================

pooled_subject_balanced_accuracy = (
    balanced_accuracy_score(
        predictions["true_label"],
        predictions["predicted_label"],
    )
)


# ============================================================
# Save results
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
    "\n"
    "========================================"
)

print(
    "FULL ROCKET REAL-DATA BASELINE"
)

print(
    "========================================"
)

print("\nPer-fold results:")

print(
    results[
        [
            "outer_fold",
            "selected_alpha",
            "window_balanced_accuracy",
            "subject_balanced_accuracy",
            "identity_accuracy",
            "identity_chance",
        ]
    ].to_string(
        index=False
    )
)

window_mean = (
    results[
        "window_balanced_accuracy"
    ].mean()
)

window_std = (
    results[
        "window_balanced_accuracy"
    ].std(
        ddof=1
    )
)

subject_mean = (
    results[
        "subject_balanced_accuracy"
    ].mean()
)

subject_std = (
    results[
        "subject_balanced_accuracy"
    ].std(
        ddof=1
    )
)

identity_mean = (
    results[
        "identity_accuracy"
    ].mean()
)

identity_std = (
    results[
        "identity_accuracy"
    ].std(
        ddof=1
    )
)

identity_chance = (
    results[
        "identity_chance"
    ].mean()
)

print(
    "\nWindow-level balanced accuracy:"
)

print(
    f"{window_mean * 100:.2f}% "
    f"± {window_std * 100:.2f}%"
)

print(
    "\nSubject-level balanced accuracy:"
)

print(
    f"{subject_mean * 100:.2f}% "
    f"± {subject_std * 100:.2f}%"
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
    f"± {identity_std * 100:.2f}%"
)

print(
    "\nIdentity chance level:"
)

print(
    f"{identity_chance * 100:.2f}%"
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