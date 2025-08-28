from sklearn.linear_model import LinearRegression

def train_linear_model(X,y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_model(X,y, model_name="linear"):
    if model_name == "linear":
        return train_linear_model(X,y)