# Coronary Heart Disease Prediction – Machine Learning Project

This project, a machine learning model for predicting coronary heart disease, was developed as part of the machine learning course at EPFL by Alexander Magnus, Eugen Bošnjak and Martin Majsec. 

To run the model, run run.py. By default, the kNN model will be run as it's the one with the best performance, but one can easily run the other models (neural network, ID3 decision tree) by toggling the `model_to_run` variable (in run.py).

Data preprocessing steps that were used in all of the models can be found separately in data_wrangling_fix_missing_data.ipynb.

During the exploratory phase, pandas was used for simplicity of data wrangling and data inspection. However, **no pandas methods were used in the final pipeline**. Instead, the column names were exported and dropped using a helper `load_csv_data` in helpers.py.

implementations.py contains implementations of ML methods seen in class i.e. everything needed for Step 2 of the project task.
helpers.py contains helper methods used in one or more model implementations i.e. everything needed for Step 3 of the project task. 

The submitted report can be seen in CS-433_Machine_Learning_Project1.pdf.

The individual model implementations are organised as follows.

## k-nearest neighbors
The model is implemented in knn.py and called from run.py. To explore the model, one can run the knn.ipynb notebook, for example by running:
```
jupyter notebook knn.ipynb
```

The non-streaming implementation of knn that was mentioned in the report and not used in the final model, can also be found in knn.py, as well as the hyperparameter tuning that was used.

## ID3
The model is implemented in ID3.py and called from run.py. It uses helpers from helpers.py. 

## Neural network
Classes used in the neural network can be found in nn.py. The model implementation can be found in nn_solution.py. The model can be run separately by running:
```
python nn_solution.py
```
