from implementations import *
from helpers import *
from ID3 import *

model_to_run = "ID3" # change this toggle to run a different model
output_file_name = "some_filename" + ".csv"

if model_to_run == "ID3":
    # Input the data
    path_to_dataset = "data/dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        path_to_dataset, NaNstrat="fill"
    ) # loading is too slow, I use max_rows and max_features for testing -M

    # set hyperparams
    k = 3 # bin size
    max_depth = 10 # max ID3 tree depth
    dim_cnt = 30 # max no of dimensions for PCA
    
    # preprocess
    # I only remove 0-variance columns (i.e. call preprocess_structural), 
    # and don't further standardize data as ID3 does not use it.
    x_train, mask = preprocess_structural(x_train, ones = False)
    x_test = x_test[:, mask] # reapply the same to test data
    # Now the common techinques: balancing the dataset and applying PCA
    x_train, y_train = balance_dataset(x_train, y_train, neg_pos_ratio = 2.0)
    x_train, x_mean, top_components = pca_fit(x_train, dim_cnt) 
    x_test = pca_transform(x_test, x_mean, top_components) # reapply the same to test data

    # ID3 specific preprocessing: convert the labels to strings... 
    y_train = y_train[:].astype(str).reshape(-1, 1)
    # ... and discretize numeric features
    bins = compute_bins(x_train, k)
    tx_disc = apply_bins(x_train, bins).astype(str) 
    x_test_disc = apply_bins(x_test, bins).astype(str) # reapply the same to test data

    # Find the best ID3 tree depth
    best_depth = test_hyperparams(tx_disc, y_train, max_depth)

    # Fit the best model we found
    # create a placeholder column because ID3::predict expects a placeholder last column
    dummy_column = np.full((tx_disc.shape[0], 1), "x") 
    final_header, final_train_data = ID3_format(tx_disc, y_train) # format the data accordingly
    best_model = ID3()
    best_model.fit(final_header, final_train_data, best_depth)

    # Generate predictions on it
    x_test_disc = apply_bins(x_test, bins)
    dummy_column_test = np.full((x_test_disc.shape[0], 1), "x") # another placeholder
    test_header, test_data = ID3_format(x_test_disc, dummy_column_test) # format the data accordingly
    predictions = best_model.predict(test_header, test_data)

    preds_to_int = [int(prediction) for prediction in predictions]
    create_csv_submission(test_ids, preds_to_int, output_file_name)