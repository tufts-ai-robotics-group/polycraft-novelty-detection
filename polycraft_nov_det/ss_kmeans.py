import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum, row_norms


# modified version of SKLearn algorithms
def ss_kmeans_plusplus(
    X, y, X_unlabeled, n_clusters_unlabled, *, x_squared_norms=None, random_state=None,
    n_local_trials=None
):
    """Init n_clusters seeds according to semi-supervised k-means++
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The labeled data to pick seeds from.
    y : {array-like, sparse matrix} of shape (n_samples,)
        The labels for X.
    X_unlabeled : {array-like, sparse matrix} of shape (n_samples, n_features)
        The unlabeled data to pick seeds from.
    n_clusters_unlabled : int
        The number of centroids to initialize from unlabeled data.
    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.
    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    """
    # Check data
    X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    X_unlabeled = check_array(X_unlabeled, accept_sparse="csr", dtype=[np.float64, np.float32])

    n_unlabeled = X_unlabeled.shape[0]
    if n_unlabeled < n_clusters_unlabled:
        raise ValueError(
            f"n_samples={n_unlabeled} should be >= n_clusters_unlabled={n_clusters_unlabled}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X_unlabeled, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X_unlabeled.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {n_unlabeled}."
        )

    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError(
            f"n_local_trials is set to {n_local_trials} but should be an "
            "integer value greater than zero."
        )

    random_state = check_random_state(random_state)

    # Call private semi-supervised k-means++
    centers = _ss_kmeans_plusplus(
        X, y, X_unlabeled, n_clusters_unlabled, x_squared_norms, random_state, n_local_trials
    )

    return centers


def _ss_kmeans_plusplus(
    X, y, X_unlabeled, n_clusters_unlabled, x_squared_norms, random_state, n_local_trials=None
):
    """Computational component for initialization of n_clusters by semi-supervised
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The labeled data to pick seeds from.
    y : {array-like, sparse matrix} of shape (n_samples,)
        The labels for X.
    X_unlabeled : {array-like, sparse matrix} of shape (n_samples, n_features)
        The unlabeled data to pick seeds from.
    n_clusters_unlabled : int
        The number of centroids to initialize from unlabeled data.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point in X_unlabeled.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    """
    _, n_features = X.shape

    centers_unlabeled = np.empty((n_clusters_unlabled, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters_unlabled))

    # Calculate centers for labeled points
    targets = np.unique(y)
    centers_labeled = np.empty((len(targets), n_features), dtype=X.dtype)
    for i, target in enumerate(targets):
        centers_labeled[i] = np.mean(X[y == target], axis=0)

    # Initialize list of closest distances and calculate current potential
    closest_dist_sqs = euclidean_distances(
        centers_labeled, X_unlabeled, Y_norm_squared=x_squared_norms, squared=True
    )
    closest_dist_sq = np.min(closest_dist_sqs, axis=0)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters_unlabled points
    for c in range(n_clusters_unlabled):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X_unlabeled[candidate_ids], X_unlabeled, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X_unlabeled):
            centers_unlabeled[c] = X_unlabeled[best_candidate].toarray()
        else:
            centers_unlabeled[c] = X_unlabeled[best_candidate]

    return np.vstack((centers_labeled, centers_unlabeled))
