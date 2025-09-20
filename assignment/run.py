from implementations import *
from helpers import load_csv_data

path_to_dataset = "data/dataset"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(path_to_dataset, max_rows=50, sub_sample=True) # I use a small sample for initial tests -M
# use the data...


