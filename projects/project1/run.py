from implementations import *
from helpers import *
from ID3 import *
from knn import *
from nn_solution import *

# change this toggle to run a different model
model_to_run = "KNN"  # options: ID3, KNN, NN
output_file_name = "some_filename" + ".csv"

if model_to_run == "ID3":
    # Input the data
    path_to_dataset = "data/dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        path_to_dataset, NaNstrat="fill"
    )

    # set hyperparams
    k = 3  # bin size
    max_depth = 10  # max ID3 tree depth
    dim_cnt = 30  # max no of dimensions for PCA

    # preprocess
    # I only remove 0-variance columns (i.e. call preprocess_structural),
    # and don't further standardize data as ID3 does not use it.
    x_train, mask = preprocess_structural(x_train, ones=False)
    x_test = x_test[:, mask]  # reapply the same to test data
    # Now the common techinques: balancing the dataset and applying PCA
    x_train, y_train = balance_dataset(x_train, y_train, neg_pos_ratio=2.0)
    x_train, x_mean, top_components = pca_fit(x_train, dim_cnt)
    x_test = pca_transform(
        x_test, x_mean, top_components
    )  # reapply the same to test data

    # ID3 specific preprocessing: convert the labels to strings...
    y_train = y_train[:].astype(str).reshape(-1, 1)
    # ... and discretize numeric features
    bins = compute_bins(x_train, k)
    tx_disc = apply_bins(x_train, bins).astype(str)
    x_test_disc = apply_bins(x_test, bins).astype(str)  # reapply the same to test data

    # Find the best ID3 tree depth
    best_depth = test_hyperparams(tx_disc, y_train, max_depth)

    # Fit the best model we found
    # create a placeholder column because ID3::predict expects a placeholder last column
    dummy_column = np.full((tx_disc.shape[0], 1), "x")
    final_header, final_train_data = ID3_format(
        tx_disc, y_train
    )  # format the data accordingly
    best_model = ID3()
    best_model.fit(final_header, final_train_data, best_depth)

    # Generate predictions on it
    x_test_disc = apply_bins(x_test, bins)
    dummy_column_test = np.full((x_test_disc.shape[0], 1), "x")  # another placeholder
    test_header, test_data = ID3_format(
        x_test_disc, dummy_column_test
    )  # format the data accordingly
    predictions = best_model.predict(test_header, test_data)

    preds_to_int = [int(prediction) for prediction in predictions]
    create_csv_submission(test_ids, preds_to_int, output_file_name)

elif model_to_run == "KNN":
    drop_columns = [
        "FRUITJU1",
        "_AIDTST3",
        "HIVTST6",
        "_FRUTSUM",
        "FRUTDA1_",
        "_FRUTSUM",
        "_BMI5",
        "HIVTST6",
        "CTELENUM",
        "PVTRESD1",
        "COLGHOUS",
        "STATERES",
        "CELLFON3",
        "LADULT",
        "NUMADULT",
        "NUMMEN",
        "NUMWOMEN",
        "CTELNUM1",
        "CELLFON2",
        "CADULT",
        "PVTRESD2",
        "CCLGHOUS",
        "CSTATE",
        "LANDLINE",
        "HHADULT",
        "POORHLTH",
        "BPMEDS",
        "ASTHNOW",
        "DIABAGE2",
        "NUMHHOL2",
        "NUMPHON2",
        "CPDEMO1",
        "PREGNANT",
        "SMOKDAY2",
        "STOPSMK2",
        "LASTSMK2",
        "AVEDRNK2",
        "DRNK3GE5",
        "MAXDRNKS",
        "EXRACT11",
        "EXEROFT1",
        "EXERHMM1",
        "EXRACT21",
        "EXEROFT2",
        "EXERHMM2",
        "LMTJOIN3",
        "ARTHDIS2",
        "ARTHSOCL",
        "JOINPAIN",
        "FLSHTMY2",
        "IMFVPLAC",
        "HIVTSTD3",
        "WHRTST10",
        "PDIABTST",
        "PREDIAB1",
        "INSULIN",
        "BLDSUGAR",
        "FEETCHK2",
        "DOCTDIAB",
        "CHKHEMO3",
        "FEETCHK",
        "EYEEXAM",
        "DIABEYE",
        "DIABEDU",
        "CAREGIV1",
        "CRGVREL1",
        "CRGVLNG1",
        "CRGVHRS1",
        "CRGVPRB1",
        "CRGVPERS",
        "CRGVHOUS",
        "CRGVMST2",
        "CRGVEXPT",
        "VIDFCLT2",
        "VIREDIF3",
        "VIPRFVS2",
        "VINOCRE2",
        "VIEYEXM2",
        "VIINSUR2",
        "VICTRCT4",
        "VIGLUMA2",
        "VIMACDG2",
        "CIMEMLOS",
        "CDHOUSE",
        "CDASSIST",
        "CDHELP",
        "CDSOCIAL",
        "CDDISCUS",
        "WTCHSALT",
        "LONGWTCH",
        "DRADVISE",
        "ASTHMAGE",
        "ASATTACK",
        "ASERVIST",
        "ASDRVIST",
        "ASRCHKUP",
        "ASACTLIM",
        "ASYMPTOM",
        "ASNOSLEP",
        "ASTHMED3",
        "ASINHALR",
        "HAREHAB1",
        "STREHAB1",
        "CVDASPRN",
        "ASPUNSAF",
        "RLIVPAIN",
        "RDUCHART",
        "RDUCSTRK",
        "ARTTODAY",
        "ARTHWGT",
        "ARTHEXER",
        "ARTHEDU",
        "TETANUS",
        "HPVADVC2",
        "HPVADSHT",
        "SHINGLE2",
        "HADMAM",
        "HOWLONG",
        "HADPAP2",
        "LASTPAP2",
        "HPVTEST",
        "HPLSTTST",
        "HADHYST2",
        "PROFEXAM",
        "LENGEXAM",
        "BLDSTOOL",
        "LSTBLDS3",
        "HADSIGM3",
        "HADSGCO1",
        "LASTSIG3",
        "PCPSAAD2",
        "PCPSADI1",
        "PCPSARE1",
        "PSATEST1",
        "PSATIME",
        "PCPSARS1",
        "PCPSADE1",
        "PCDMDECN",
        "SCNTMNY1",
        "SCNTMEL1",
        "SCNTPAID",
        "SCNTWRK1",
        "SCNTLPAD",
        "SCNTLWK1",
        "SXORIENT",
        "TRNSGNDR",
        "RCSGENDR",
        "RCSRLTN2",
        "CASTHDX2",
        "CASTHNO2",
        "EMTSUPRT",
        "LSATISFY",
        "ADPLEASR",
        "ADDOWN",
        "ADSLEEP",
        "ADENERGY",
        "ADEAT1",
        "ADFAIL",
        "ADTHINK",
        "ADMOVE",
        "MISTMNT",
        "ADANXEV",
        "MSCODE",
        "_CRACE1",
        "_CPRACE",
        "_CLLCPWT",
        "_DUALCOR",
        "METVL11_",
        "METVL21_",
        "ACTIN11_",
        "ACTIN21_",
        "PADUR1_",
        "PADUR2_",
        "PAFREQ1_",
        "PAFREQ2_",
        "_MINAC11",
        "_MINAC21",
        "PAMIN11_",
        "PAMIN21_",
        "PA1MIN_",
        "PAVIG11_",
        "PAVIG21_",
        "PA1VIGM_",
        "_FLSHOT6",
        "_PNEUMO2",
    ]
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        "data/dataset", max_rows=None, NaNstrat="fill", remove_columns=None
    )

    tx, mask = preprocess_structural(x_train, ones=False)
    tx = preprocess_unstructural(tx)

    x_test = x_test[:, mask]
    x_test = preprocess_unstructural(x_test)

    tx_reduced, X_mean, top_components = pca_fit(tx, 50)
    x_test_reduced = pca_transform(x_test, X_mean, top_components)

    tx_train_split, y_train_split = tx_reduced, y_train
    tx_train_split_balanced, y_train_split_balanced = balance_dataset(
        tx_train_split, y_train_split, 2
    )
    y_train_split_balanced = np.where(
        y_train_split_balanced == -1, 0, y_train_split_balanced
    )
    predictions = knn_predict_streaming(
        tx_train_split_balanced, y_train_split_balanced, x_test_reduced, 17
    )
    predictions = np.where(predictions == 0, -1, predictions)
    create_csv_submission(test_ids, predictions, output_file_name)

elif model_to_run == "NN":
    # Load the data.
    PATH_TO_DATASET = "data/dataset"
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        PATH_TO_DATASET, NaNstrat="fill", remove_columns=DROP_COLUMNS, max_rows=1000
    )

    # Take only a prefix of the rows, it's too slow with the entire dataset.
    MAX_ROWS = 70000
    x_train = x_train[:MAX_ROWS]
    y_train = y_train[:MAX_ROWS]

    # Balance the dataset because no. of 1s is way smaller than no. of 0s.
    x_train, y_train = balance_dataset(x_train, y_train, 2)

    # Preprocess the train dataset.
    x_train, mask = preprocess_structural(x_train, ones=False)
    x_train = preprocess_unstructural(x_train)

    # Preprocess the test dataset.
    x_test = x_test[:, mask]
    x_test = preprocess_unstructural(x_test)

    # Apply dimensionality reduction.
    MAX_PCA_DIMS = 35
    x_train, x_mean, top_comps = pca_fit(x_train, MAX_PCA_DIMS)
    x_test = pca_transform(x_test, x_mean, top_comps)

    # Split the training dataset for evaluation.
    tx_train_split, tx_test_split, y_train_split, y_test_split = split_data(
        x_train, y_train
    )

    y_train_split = y_train_split.reshape((-1, 1))
    y_test_split = y_test_split.reshape((-1, 1))
    y_train_split = (1 + y_train_split) / 2
    y_test_split = (1 + y_test_split) / 2

    print(np.shape(tx_train_split))
    print(np.shape(y_train_split))

    D = tx_train_split.shape[1]
    NUM_ITER = 1000

    # Run the genetic algorithm.
    nn_shape = [D, 8, 8, 8, 8, 1]
    w = gen_alg(tx_train_split, y_train_split, nn_shape, 16, 1, 0.1, 0.1, NUM_ITER)

    # Calculate the confusion matrix and graph it.
    y_pred = (NN(nn_shape, w).evaluate(tx_test_split) > 0.5).astype(np.float128)
    print(f"predicted {y_pred[:16]}")
    print(f"true {y_test_split[:16]}")

    # Evaluate on actual test dataset for submission.
    y_submit = (NN(nn_shape, w).evaluate(x_test) > 0.5).astype(np.int32) * 2 - 1
    create_csv_submission(test_ids, y_submit, name=output_file_name)
