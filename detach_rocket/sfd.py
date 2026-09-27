"""Sequential Feature Detachment (SFD) core methods."""

import numpy as np

from detach_rocket._warnings import quiet_ridge_warnings


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    """
    L2-normalize a feature-importance vector.

    If the vector has zero norm, return a vector of zeros.
    """
    values = np.asarray(values, dtype=float)

    norm = np.linalg.norm(values, ord=2)

    if norm == 0:
        return np.zeros_like(values)

    return values / norm


def feature_detachment(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    drop_ratio: float = 0.05,
    num_steps: int = 150,
    multiclass_type: str = "max",
    verbose: bool = False,
    identity_classifier=None,
    y_identity: np.ndarray = None,
    identity_lambda: float = 0.0,
):
    """
    Apply Sequential Feature Detachment (SFD) to a feature matrix.

    The function supports two modes:

    1. Standard SFD
       Features are ranked only according to their importance
       for predicting the primary task.

    2. Identity-aware SFD
       Features are ranked according to:

           score_j =
               normalized_task_importance_j
               - identity_lambda
               * normalized_identity_importance_j

       where identity importance is measured using the maximum
       absolute coefficient across subject classes.

    Parameters
    ----------
    classifier : sklearn model
        Ridge linear classifier for the primary task.

    X_train : numpy array
        Training feature matrix with shape
        (n_instances, n_features).

    y_train : numpy array
        Primary task labels.

    X_test : numpy array
        Validation/test feature matrix.

    y_test : numpy array
        Validation/test primary task labels.

    drop_ratio : float
        Proportion of features dropped at each SFD step.

    num_steps : int
        Maximum number of detachment steps.

    multiclass_type : str
        Method used to aggregate coefficients for a multiclass
        primary task.

        Options:

            "max"
            "norm"
            "avg"

    verbose : bool
        If True, print pruning progress.

    identity_classifier : sklearn model or None
        Auxiliary Ridge classifier used to predict subject
        identity.

        If None, standard SFD is used.

    y_identity : numpy array or None
        Subject-identity labels for the training recordings.

        Required when identity_classifier is supplied.

    identity_lambda : float
        Strength of the subject-identity penalty.

        identity_lambda = 0 reproduces standard task-based
        feature ordering because L2 normalization preserves
        the ordering of task importance values.

    Returns
    -------
    retained_ratios : numpy array
        Proportion of features retained at each detachment step.

    score_list_train : numpy array
        Primary-task training accuracy at each step.

    score_list_test : numpy array or None
        Primary-task validation/test accuracy at each step.

    feature_importance_matrix : numpy array
        Feature-ranking information at each SFD step.

        For standard SFD, this contains the normal feature
        importance values.

        For identity-aware SFD, selected feature scores are
        shifted to positive values before storage so that the
        existing Detach-ROCKET feature-mask logic remains
        compatible. The actual feature selection still uses
        the unshifted identity-aware score.
    """

    # =====================================================
    # Validate inputs
    # =====================================================

    if y_train is None:
        raise ValueError(
            "y_train must be provided."
        )

    if not 0 < drop_ratio < 1:
        raise ValueError(
            "drop_ratio must be between 0 and 1."
        )

    if num_steps <= 0:
        raise ValueError(
            "num_steps must be greater than 0."
        )

    if identity_lambda < 0:
        raise ValueError(
            "identity_lambda must be non-negative."
        )


    identity_aware = (
        identity_classifier is not None
        or y_identity is not None
    )


    if identity_aware:

        if identity_classifier is None:
            raise ValueError(
                "identity_classifier must be provided "
                "when using identity-aware SFD."
            )

        if y_identity is None:
            raise ValueError(
                "y_identity must be provided "
                "when using identity-aware SFD."
            )

        if len(y_identity) != len(y_train):
            raise ValueError(
                "y_identity and y_train must contain "
                "the same number of samples."
            )


    # =====================================================
    # Fit initial task classifier if necessary
    # =====================================================

    if not hasattr(classifier, "coef_"):

        with quiet_ridge_warnings():

            classifier.fit(
                X_train,
                y_train,
            )


    # =====================================================
    # Fit initial identity classifier if necessary
    # =====================================================

    if identity_aware:

        if not hasattr(
            identity_classifier,
            "coef_",
        ):

            with quiet_ridge_warnings():

                identity_classifier.fit(
                    X_train,
                    y_identity,
                )


    # =====================================================
    # Feature-importance functions
    # =====================================================

    multiclass_methods = {

        "norm": lambda coef: np.linalg.norm(
            coef,
            axis=0,
            ord=2,
        ),

        "max": lambda coef: np.linalg.norm(
            coef,
            axis=0,
            ord=np.inf,
        ),

        "avg": lambda coef: np.linalg.norm(
            coef,
            axis=0,
            ord=1,
        ),
    }


    # -----------------------------------------------------
    # Primary-task importance
    # -----------------------------------------------------

    if len(np.shape(classifier.coef_)) > 1:

        if multiclass_type not in multiclass_methods:

            raise ValueError(
                'Invalid multiclass_type. '
                'Choose from: "norm", "max", or "avg".'
            )

        calc_task_importance = (
            multiclass_methods[
                multiclass_type
            ]
        )

    else:

        def calc_task_importance(coef):

            return np.abs(
                coef
            ).ravel()


    # -----------------------------------------------------
    # Identity importance
    # -----------------------------------------------------

    def calc_identity_importance(coef):
        """
        For multiclass subject identity, use the maximum
        absolute coefficient across subjects.

        This corresponds to:

            I_identity,j =
                max_s |beta_s,j|
        """

        coef = np.asarray(coef)

        if coef.ndim == 1:

            return np.abs(
                coef
            ).ravel()

        return np.max(
            np.abs(coef),
            axis=0,
        )


    # =====================================================
    # Determine feature counts for SFD
    # =====================================================

    total_features = X_train.shape[1]

    retain_ratio = (
        1 - drop_ratio
    )


    retained_ratios_uniform = np.power(
        retain_ratio,
        np.arange(num_steps),
    )


    retained_features = np.unique(
        (
            retained_ratios_uniform
            * total_features
        ).astype(int)
    )


    retained_features = (
        retained_features[::-1]
    )


    retained_features = retained_features[
        retained_features > 0
    ]


    retained_ratios = (
        retained_features
        / total_features
    )


    # =====================================================
    # Storage
    # =====================================================

    score_list_train = []


    score_list_test = (
        []
        if X_test is not None
        and y_test is not None
        else None
    )


    selection_mask = np.full(
        total_features,
        False,
    )


    feature_importance_matrix = np.zeros(
        (
            len(retained_features),
            total_features,
        )
    )


    # =====================================================
    # Initial feature importance
    # =====================================================

    task_importance = (
        calc_task_importance(
            classifier.coef_
        )
    )


    if identity_aware:

        identity_importance = (
            calc_identity_importance(
                identity_classifier.coef_
            )
        )


        normalized_task = (
            _l2_normalize(
                task_importance
            )
        )


        normalized_identity = (
            _l2_normalize(
                identity_importance
            )
        )


        ranking_score = (
            normalized_task
            - identity_lambda
            * normalized_identity
        )


    else:

        # Preserve the original standard SFD behavior.
        feature_importance = (
            task_importance.copy()
        )


    # =====================================================
    # Sequential Feature Detachment
    # =====================================================

    for count, num_features in enumerate(
        retained_features
    ):

        # -------------------------------------------------
        # Select features
        # -------------------------------------------------

        if identity_aware:

            selected_idxs = np.argsort(
                ranking_score
            )[-num_features:]

        else:

            selected_idxs = np.argsort(
                feature_importance
            )[-num_features:]


        selection_mask[:] = False

        selection_mask[
            selected_idxs
        ] = True


        X_train_subsampled = X_train[
            :,
            selection_mask,
        ]


        # -------------------------------------------------
        # Retrain primary task classifier
        # -------------------------------------------------

        with quiet_ridge_warnings():

            classifier.fit(
                X_train_subsampled,
                y_train,
            )


        avg_score_train = classifier.score(
            X_train_subsampled,
            y_train,
        )


        score_list_train.append(
            avg_score_train
        )


        # -------------------------------------------------
        # Evaluate primary task on validation data
        # -------------------------------------------------

        if score_list_test is not None:

            X_test_subsampled = X_test[
                :,
                selection_mask,
            ]


            avg_score_test = classifier.score(
                X_test_subsampled,
                y_test,
            )


            score_list_test.append(
                avg_score_test
            )


        # =================================================
        # Standard SFD update
        # =================================================

        if not identity_aware:

            feature_importance[
                ~selection_mask
            ] = 0


            feature_importance_matrix[
                count,
                :
            ] = feature_importance


            feature_importance[
                selection_mask
            ] = calc_task_importance(
                classifier.coef_
            )


        # =================================================
        # Identity-aware SFD update
        # =================================================

        else:

            # ---------------------------------------------
            # Store the current selected feature ranking
            # ---------------------------------------------
            #
            # Existing DetachRocket code later constructs
            # the selected feature mask using:
            #
            #     importance_matrix_[step] > 0
            #
            # Identity-aware scores can legitimately be
            # negative, so we shift the selected scores
            # above zero ONLY for storage.
            #
            # This shift DOES NOT affect feature ranking.
            # ---------------------------------------------

            stored_scores = np.zeros(
                total_features,
                dtype=float,
            )


            current_scores = ranking_score[
                selection_mask
            ]


            if len(current_scores) > 0:

                minimum_score = np.min(
                    current_scores
                )


                shifted_scores = (
                    current_scores
                    - minimum_score
                    + np.finfo(float).eps
                )


                stored_scores[
                    selection_mask
                ] = shifted_scores


            feature_importance_matrix[
                count,
                :
            ] = stored_scores


            # ---------------------------------------------
            # Retrain subject-identity classifier
            # ---------------------------------------------

            with quiet_ridge_warnings():

                identity_classifier.fit(
                    X_train_subsampled,
                    y_identity,
                )


            # ---------------------------------------------
            # Recalculate task importance
            # ---------------------------------------------

            task_importance_selected = (
                calc_task_importance(
                    classifier.coef_
                )
            )


            # ---------------------------------------------
            # Recalculate identity importance
            # ---------------------------------------------

            identity_importance_selected = (
                calc_identity_importance(
                    identity_classifier.coef_
                )
            )


            # ---------------------------------------------
            # L2-normalize independently
            # ---------------------------------------------

            normalized_task_selected = (
                _l2_normalize(
                    task_importance_selected
                )
            )


            normalized_identity_selected = (
                _l2_normalize(
                    identity_importance_selected
                )
            )


            # ---------------------------------------------
            # Identity-aware ranking
            # ---------------------------------------------

            updated_scores = (
                normalized_task_selected
                - identity_lambda
                * normalized_identity_selected
            )


            # ---------------------------------------------
            # Detached features must never re-enter
            # ---------------------------------------------

            ranking_score[:] = -np.inf


            ranking_score[
                selection_mask
            ] = updated_scores


        # -------------------------------------------------
        # Progress output
        # -------------------------------------------------

        if verbose:

            current_percentage = (
                retained_ratios[count]
                * 100
            )


            if identity_aware:

                print(
                    f"Step {count + 1} out of "
                    f"{len(retained_features)}: "
                    f"{current_percentage:.2f}% "
                    f"of features used "
                    f"| identity lambda = "
                    f"{identity_lambda}"
                )

            else:

                print(
                    f"Step {count + 1} out of "
                    f"{len(retained_features)}: "
                    f"{current_percentage:.2f}% "
                    f"of features used"
                )


    # =====================================================
    # Convert results to NumPy arrays
    # =====================================================

    if score_list_test is not None:

        score_list_test = np.asarray(
            score_list_test
        )


    return (
        retained_ratios,
        np.asarray(
            score_list_train
        ),
        score_list_test,
        feature_importance_matrix,
    )