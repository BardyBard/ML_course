from implementations import *
from helpers import load_csv_data

path_to_dataset = "data/dataset"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
   path_to_dataset, max_rows=10, max_features = 5, NaNstrat="fill") # I use a small sample for initial tests -M
# now use the data...

debug = False # turn this on to see verbose logs

if debug:
   print("x ", x_train)
   print("y ", y_train)

# preprocess data -> will be moved to a separate method later on

#####
# standardize
x_mean = np.nanmean(x_train, axis=0)
x_std = np.nanstd(x_train, axis=0)
x_train_standardized = (x_train - x_mean) / x_std
# clip extreme outliers
x_train = np.clip(x_train_standardized, -5, 5)
# prepend the 1s column 
ones = np.ones((len(x_train),1), dtype=float)
tx = np.hstack((ones, x_train.astype(float)))
if debug : print("tx ", tx)
#####

# try some parameters
w0 = np.zeros(tx.shape[1])
max_iters = 500
gamma = 0.1 # empirically this value seems to work alright -M

# some debug info
if debug:
   grad = compute_gradient(y_train, tx, w0)
   print("Initial gradient norm:", np.linalg.norm(grad))

# Run it!
w, loss1 = mean_squared_error_gd(y_train, tx, w0, max_iters, gamma)
w, loss2 = mean_squared_error_sgd(y_train, tx, w0, max_iters, gamma)
w, loss3 = least_squares(y_train, tx) 
w, loss4 = ridge_regression(y_train, tx, .1)

if debug : print(w, sep = "\n----------------------\n")
# Print the results
print(f"Initial weights: {w0}")
print(f"No of iterations: {max_iters}")
print(f"Learning rate: {gamma}")
print(f"L[GD] = {loss1}", f"L[SGD] = {loss2}", f"L[OLS] = {loss3}", 
      f"L[RR] = {loss4}", sep = "\n")