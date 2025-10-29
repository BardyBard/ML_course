from implementations import *
from helpers import load_csv_data, print_result

path_to_dataset = "data/dataset"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    path_to_dataset, max_rows=500, max_features=None, NaNstrat="fill"
)  # I use a small sample for initial tests -M
# now use the data...

debug = False  # turn this on to see verbose logs

if debug:
    print("x ", x_train)
    print("y ", y_train)

tx, mask = preprocess(x_train)
# remove 0-variance in test data too
x_test = x_test[:, mask]

# try some parameters
w0 = np.zeros(tx.shape[1])
max_iters = 1000
gamma = 0.1  # empirically these gamma values seems to work alright -M
gamma_sgd = 0.001
gamma_logistic = 0.001
lambda_ = 0.1  # I guess the regularization param in ridge doesn't have to be very big for standardized data -M
lambda_LR = 0.1

if debug:  # some debug info
    grad = compute_gradient(y_train, tx, w0)
    print("Initial gradient norm:", np.linalg.norm(grad))

# Run it!
print("3shapes", y_train.shape, tx.shape, w0.shape)

w1, loss1 = mean_squared_error_gd(y_train, tx, w0, max_iters, gamma)
# w2, loss2 = mean_squared_error_sgd(y_train, tx, w0, max_iters, gamma_sgd)
# w3, loss3 = least_squares(y_train, tx)
# w4, loss4 = ridge_regression(y_train, tx, lambda_)
# w5, loss5 = logistic_regression(y_train, tx, w0, max_iters, gamma_logistic)
# w6, loss6 = reg_logistic_regression(
#     y_train, tx, lambda_LR, w0, max_iters, gamma_logistic
# )


# if debug : print(, sep = "\n----------------------\n")
# Print the results
# print(f"Initial weights: {w0}")
print(f"No of iterations: {max_iters}")
print(f"Learning rate (GD): {gamma}")
print(f"Learning rate (SGD): {gamma_sgd}")
print(f"Learning rate (LR): {gamma_logistic}")
print(f"Regularization term (RR): {lambda_}")
print(f"Regularization term (LR-reg): {lambda_LR}")

print_result("GD", loss1, w1)
# print_result("SGD", loss2, w2)
# print_result("OLS", loss3, w3)
# print_result("RR", loss4, w4)
# print_result("LR", loss5, w5, f"gamma = {gamma_logistic}")
# print_result("LR-reg", loss6, w6)

# training is done. Onto testing!
print("2shapes", w1.shape, x_test.shape)
print(x_test)
y_pred_binary = np.sign(x_test @ w1[1:])
print(y_pred_binary)
