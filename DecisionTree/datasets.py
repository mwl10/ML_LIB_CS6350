import pandas as pd
import numpy as np

def get_bank_data(train_fp='../datasets/bank/train.csv',
                  test_fp='../datasets/bank/test.csv'):
    train_data = pd.read_csv(train_fp, header=None).to_numpy()
    test_data = pd.read_csv(test_fp, header=None).to_numpy()
    feat_names = np.array(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
           'loan', 'contact', 'day_of_week', 'month', 'duration', 'campaign',
           'pdays', 'previous', 'poutcome'])
    X_train, y_train = train_data[:, :-1], train_data[:, -1:]
    X_test, y_test = test_data[:, :-1], test_data[:, -1:]
    
    # converting numerical features to binary ones using media
    numeric_features = [0,5,9,11,12,13,14]
    medians = np.median(X_train[:, numeric_features], axis=0)
    X_train[:,numeric_features] = X_train[:,numeric_features] > medians
    X_test[:,numeric_features] = X_test[:,numeric_features] > medians

    return (X_train, y_train), (X_test, y_test), feat_names
    