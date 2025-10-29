# ---------------- KNN IMPLEMENTATION
# Built with the help of https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def get_neighbors(train, test_row, num_neighbors):
    """Find k nearest neighbors using vectorized operations.

    Args:
        train: numpy array of training data
        test_row: single test instance
        num_neighbors: number of neighbors to return

    Returns:
        numpy array of num_neighbors nearest training instances
    """
    # Vectorized distance computation
    distances = np.sqrt(np.sum((train[:, :-1] - test_row[:-1]) ** 2, axis=1))

    # Get indices of k nearest neighbors
    nearest_neighbor_indices = np.argsort(distances)[:num_neighbors]

    # Return the k nearest neighbors
    return train[nearest_neighbor_indices]


def predict_classification_batch(train_features, train_labels, test_features, num_neighbors):
    """Make classification predictions for multiple test instances using vectorized operations.

    Args:
        train_features: numpy array of training features (N_train, D)
        train_labels: numpy array of training labels (N_train,)
        test_features: numpy array of test features (N_test, D)
        num_neighbors: number of neighbors to use

    Returns:
        numpy array of shape (N_test,) containing predictions for all test instances
    """
    # Compute distances for all test samples at once
    # Shape: (N_test, N_train)
    # Using broadcasting: test_features[:, np.newaxis, :] creates shape (N_test, 1, D)
    # train_features creates shape (N_train, D) and broadcasts to (N_test, N_train, D)
    distances = np.sqrt(
        np.sum((test_features[:, np.newaxis, :] - train_features) ** 2, axis=2))

    # Get indices of k nearest neighbors for each test sample
    # Shape: (N_test, num_neighbors)
    nearest_indices = np.argsort(distances, axis=1)[:, :num_neighbors]

    # Get the labels of the k nearest neighbors
    # Shape: (N_test, num_neighbors)
    neighbor_labels = train_labels[nearest_indices]

    # For each test sample, find the most common label among its k neighbors
    predictions = np.array([
        np.bincount(neighbor_labels[i].astype(int)).argmax()
        for i in range(len(test_features))
    ])

    return predictions

def knn_predict_streaming(train_X, train_y, test_X, k,
                          test_batch=1024, train_block=4096,
                          as_float32=True):
    """
    Memory-efficient exact KNN classification.
    - No (N_test x N_train x D) broadcast
    - Processes test and train in blocks
    - Uses squared distances via norms + GEMM
    """
    if as_float32:
        train_X = np.asarray(train_X, dtype=np.float32, order="C")
        test_X  = np.asarray(test_X,  dtype=np.float32, order="C")
    else:
        train_X = np.ascontiguousarray(train_X)
        test_X  = np.ascontiguousarray(test_X)

    # ensure labels are ints for bincount
    train_y = np.asarray(train_y)
    if not np.issubdtype(train_y.dtype, np.integer):
        # map arbitrary labels to 0..C-1
        uniq, inv = np.unique(train_y, return_inverse=True)
        y_int = inv.astype(np.int64)
        to_label = uniq
    else:
        y_int = train_y.astype(np.int64, copy=False)
        to_label = None

    N_train = train_X.shape[0]
    N_test  = test_X.shape[0]
    k = int(k)

    # Precompute train squared norms once
    train_norm = np.einsum('ij,ij->i', train_X, train_X)
    preds = np.empty(N_test, dtype=np.int64)

    for t0 in range(0, N_test, test_batch):
        t1 = min(t0 + test_batch, N_test)
        Xb = test_X[t0:t1]                                  # (B, D)
        B = Xb.shape[0]
        Xb_norm = np.einsum('ij,ij->i', Xb, Xb)             # (B,)

        # running top-k (best distances and indices) per test row
        best_d = np.full((B, k), np.inf, dtype=Xb.dtype)
        best_i = np.full((B, k), -1, dtype=np.int64)

        for s0 in range(0, N_train, train_block):
            s1 = min(s0 + train_block, N_train)
            Yb = train_X[s0:s1]                              # (T, D)
            Yb_norm = train_norm[s0:s1]                      # (T,)

            # squared distances via ||x||^2 + ||y||^2 - 2 x·y
            # allocates only a (B x T) matrix, no third dim
            cross = Xb @ Yb.T                                # (B, T)
            d2 = Xb_norm[:, None] + Yb_norm[None, :] - 2.0 * cross

            # local top-k per row in this block
            # (argpartition is O(n) and avoids full sort)
            if d2.shape[1] > k:
                part_idx = np.argpartition(d2, k-1, axis=1)[:, :k]
            else:
                part_idx = np.arange(d2.shape[1])[None, :].repeat(B, 0)
            local_d = np.take_along_axis(d2, part_idx, axis=1)
            local_i = part_idx + s0

            # merge current best with local best (size 2k) and keep k
            merged_d = np.concatenate([best_d, local_d], axis=1)
            merged_i = np.concatenate([best_i, local_i], axis=1)
            sel = np.argpartition(merged_d, k-1, axis=1)[:, :k]
            best_d = np.take_along_axis(merged_d, sel, axis=1)
            best_i = np.take_along_axis(merged_i, sel, axis=1)

        # vote among k neighbors
        neigh_labels = y_int[best_i]                         # (B, k)
        # bincount per row (works when labels are small non-negative ints)
        for r in range(B):
            preds[t0 + r] = np.bincount(neigh_labels[r]).argmax()

    # map back to original labels if needed
    if to_label is not None:
        return to_label[preds]
    return preds