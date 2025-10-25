"""Some helper functions for project 1."""

import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# I added max_rows to speed up the loading.
# It was taking me way too long to load the whole sample and subsample it later.
# That can be removed when we're done testing. -M
def load_csv_data(
    data_path, max_rows=None, max_features=None, NaNstrat=None, sub_sample=False
):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
        max_rows=max_rows,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"),
        delimiter=",",
        skip_header=1,
        max_rows=max_rows,
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"),
        delimiter=",",
        skip_header=1,
        max_rows=max_rows,
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # remove duplicate rows

    # sub-sample
    if sub_sample:  # unused
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    if max_features:
        x_train = x_train[:, :max_features]

    if max_rows:
        y_train = y_train[:max_rows]
        x_train = x_train[:max_rows]
        train_ids = train_ids[:max_rows]

    if NaNstrat:
        x_train = removeNaNs(x_train)
        x_test = removeNaNs(x_test)
        
    # print("shapes", x_train.shape, x_test.shape, y_train.shape)
    # assert x_train.shape == x_test.shape
    return x_train, x_test, y_train, train_ids, test_ids

def removeNaNs(x):
    """
    x (np.array): data from which NaNs should be removed
    """
    # NaNcols = ~np.all(np.isnan(x), axis=0)
    # x = x[:, NaNcols]
    x[:, np.all(np.isnan(x), axis=0)] = 0
    # temporary solution: fill NaNs with column mean
    col_means = np.nanmean(x, axis=0)
    NaNrows = np.where(np.isnan(x))
    x[NaNrows] = np.take(col_means, NaNrows[1])
    return x


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def print_result(method_name, loss, w, additional_info=None):
    """
    This function formats and prints the outputs of a regression method.
    Loss is rounded to 5 decimals, and weights to 2.
    If `additional_info` is provided, it will be printed at the end of the output.

    Args:
        method_name (str): name of the method used
        loss (float): loss returned by the regression method
        w (list, np.array): weights returned by the regression method
        additional_info (str): optional information that could be printed at the end of the output
    """
    out = f"[{method_name}] loss = {loss:.5f}, w = {np.round(w, 2)}"
    if additional_info:
        out += f" ({additional_info})"
    print(out)


def split_data(x, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.

    Args:
        x (np.array): Feature matrix
        y (np.array): Target vector
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Controls the shuffling applied to the data

    Returns:
        x_train (np.array): Training data
        x_test (np.array): Test data
        y_train (np.array): Training labels
        y_test (np.array): Test labels
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size should be between 0 and 1")

    if len(y.shape) != 1:
        raise ValueError("y should be a 1-dimensional array")

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y should have same number of rows")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(test_size * n_samples)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def create_confusion_matrix(y_test_split, y_pred_binary):
    # Create confusion matrix manually
    # True Negatives: actual -1, predicted -1
    tn = np.sum((y_test_split == -1) & (y_pred_binary == -1))
    # False Positives: actual -1, predicted 1
    fp = np.sum((y_test_split == -1) & (y_pred_binary == 1))
    # False Negatives: actual 1, predicted -1
    fn = np.sum((y_test_split == 1) & (y_pred_binary == -1))
    # True Positives: actual 1, predicted 1
    tp = np.sum((y_test_split == 1) & (y_pred_binary == 1))

    return np.array([[tn, fp],
                   [fn, tp]])

def cm_visualization(cm):
    """
    Create and return a visualization of a confusion matrix.

    Args:
        cm (np.array): 2x2 confusion matrix array containing [TN, FP], [FN, TP]

    Returns:
        matplotlib.figure.Figure: The generated confusion matrix visualization
    """
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100

    # Create annotations with both counts and percentages
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'

    # Plot confusion matrix
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['Predicted -1', 'Predicted 1'],
                yticklabels=['Actual -1', 'Actual 1'])
    plt.title('Confusion Matrix')
        plot_title += title
    plt.title(plot_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig

def pca_reduction(X, k):
    """
    Perform Principal Component Analysis (PCA) dimensionality reduction on the input data.

    Args:
        X (np.array): Input data matrix of shape (n_samples × n_features)
        k (int): Number of principal components to keep

    Returns:
        np.array: Transformed data matrix of shape (n_samples × k) projected onto
                 the first k principal components
    """
    # X: your data as a NumPy array (n_samples × n_features)
    # 1. Center the data
    X_centered = X - np.mean(X, axis=0)

    # 2. Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 3. Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort eigenvalues (and eigenvectors) in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_components = eigenvectors[:, :k]

    X_pca = np.dot(X_centered, top_components)

    return X_pca