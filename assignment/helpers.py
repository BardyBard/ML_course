"""Some helper functions for project 1."""

import csv
import numpy as np
import os

# I added max_rows to speed up the loading. 
# It was taking me way too long to load the whole sample and subsample it later.
# That can be removed when we're done testing. -M
def load_csv_data(data_path, max_rows = None, max_features = None, NaNstrat = None, sub_sample=False):
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
        max_rows=max_rows
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1, max_rows=max_rows
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1, max_rows=max_rows
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # remove duplicate rows

    # sub-sample
    if sub_sample: # unused
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
        # remove all columns that contain only NaNs
        NaNcols = ~np.all(np.isnan(x_train), axis=0)
        x_train = x_train[:, NaNcols]
        # temporary solution: fill NaNs with column mean
        col_means = np.nanmean(x_train, axis=0)
        NaNrows = np.where(np.isnan(x_train))
        x_train[NaNrows] = np.take(col_means, NaNrows[1])


    return x_train, x_test, y_train, train_ids, test_ids


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

def print_result(method_name, loss, w, additional_info = None):
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