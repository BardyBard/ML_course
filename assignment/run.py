from implementations import *
from helpers import load_csv_data, print_result

path_to_dataset = "data/dataset"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
   path_to_dataset, max_rows=60, max_features = 50, NaNstrat="fill") # I use a small sample for initial tests -M
# now use the data...

DebuggedOLS = False # turn this on when least squares is fixed
debug = False # turn this on to see verbose logs

if debug:
   print("x ", x_train)
   print("y ", y_train)

tx = preprocess(x_train)

# try some parameters
w0 = np.zeros(tx.shape[1])
max_iters = 100
gamma = .1 # empirically these gamma values seems to work alright -M
gamma_logistic = .001
lambda_ = .1 # I guess the regularization param in ridge doesn't have to be very big for standardized data -M

if debug: # some debug info
   grad = compute_gradient(y_train, tx, w0)
   print("Initial gradient norm:", np.linalg.norm(grad))

# Run it!
w1, loss1 = mean_squared_error_gd(y_train, tx, w0, max_iters, gamma)
w2, loss2 = mean_squared_error_sgd(y_train, tx, w0, max_iters, gamma) # blows up with lots of iterations
if DebuggedOLS : w3, loss3 = least_squares(y_train, tx) 
w4, loss4 = ridge_regression(y_train, tx, lambda_) 
w5, loss5 = logistic_error_gd(y_train, tx, w0, max_iters, gamma_logistic)


# if debug : print(, sep = "\n----------------------\n")
# Print the results
print(f"Initial weights: {w0}")
print(f"No of iterations: {max_iters}")
print(f"Learning rate: {gamma}")


print_result("GD", loss1, w1)
print_result("SGD", loss2, w2) 
if DebuggedOLS : print_result("OLS", loss3, w3)
print_result("RR", loss4, w4)
print_result("LR", loss5, w5, f"gamma = {gamma_logistic}")