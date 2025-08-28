import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data import data_pipeline
from model import train_model

def get_train_val_split(X, y,shuffle=True, split = 0.2):
    if shuffle:
        idxs = np.random.shuffle(np.linspace(0,len(X),len(X)))
        X = X[idxs]
        y = y[idxs]
    X_train, X_val, y_train, y_val =  train_test_split(X,y,test_size=split)
    return X_train, X_val, y_train, y_val


def training_pipeline(data_file_path):
    X, y = data_pipeline(data_file_path)
    X_train, X_val, y_train, y_val = get_train_val_split(X,y)
    model = train_model(X_train,y_train,model_name="linear")
    y_val_predict = model.predict(X_val)
    metric = mean_squared_error(y_val,y_val_predict)
    print(f"Validation MSE: {metric}")


if __name__ == "__main__":
    csv_path = os.path.join('data', 'train_v9rqX0R.csv')
    training_pipeline(csv_path)
    



