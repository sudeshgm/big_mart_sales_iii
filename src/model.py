from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_linear_model(X,y):
    """Train with LinearRegressor

    Args:
        X (np.Ndarray): Features
        y (np.Ndarray): values to predict

    Returns:
        LinearRegression: Trained Model
    """    
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_random_forest_model(X,y):
    """Train with RandomForestRegressor

    Args:
        X (np.Ndarray): Features
        y (np.Ndarray): values to predict

    Returns:
        RandomForestRegressor: Trained Model
    """    
    model = RandomForestRegressor()
    model.fit(X,y)
    return model

def train_model(X,y, model_name="linear"):
    """Select between multiple models

    Args:
        X (np.Ndarray): Features
        y (np.Ndarray): values to predict
        model_name (str, optional): Name of the model. Defaults to "linear".

    Returns:
        _type_: trained model
    """
    if model_name == "linear":
        return train_linear_model(X,y)
    if model_name =="random_forest":
        return train_random_forest_model(X,y)