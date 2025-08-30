import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from literals import CATEGORICAL_COLS, CLASSES, FAT_MAP, MIN_MAX, NORMALIZE_COLS

def read_data(path):
    """Read csv dataset

    Args:
        path (str): Path to file

    Returns:
        pd.DataFrame: Dataset
    """    
    return pd.read_csv(path)

def split_identifier(data):
    """#1 Handles preprocessing of column Item_Identifier

    Args:
        data (pd.DataFrame): dataset
    """    
    data["Item_Identifier_1"] = data["Item_Identifier"].str[0:2]
    data["Item_Identifier_2"] = data["Item_Identifier"].str[2:3]
    data["Item_Identifier_3"] = data["Item_Identifier"].str[3:]

def map_fat(data):
    """#3 Handles preprocessing of column Item_Fat_Content

    Args:
        data (pd.DataFrame): dataset
    """    
    for key in FAT_MAP.keys():
        data["Item_Fat_Content"][data["Item_Fat_Content"] == key] = FAT_MAP[key]

def set_outlet_size(data):
    """#9 Handles proprocessing of column Outlet_Size

    Args:
        data (pd.DataFrame): Dataset
    """
    ['Small','Medium','High'],
    data["Outlet_Size"][data["Outlet_Size"]=='Small'] = 1
    data["Outlet_Size"][data["Outlet_Size"]=='Medium'] = 2
    data["Outlet_Size"][data["Outlet_Size"]=='High'] = 3

def extract_tier_level(data):
    """#10 Handles proprocessing of column Outlet_Location_Type

    Args:
        data (pd.DataFrame): Dataset
    """    
    data["Outlet_Location_Type"] = int(data["Outlet_Location_Type"].str[-1])

def extract_supermarket_level(data):
    """#11 Handles preprocessing of Outlet_Location_Type

    Args:
        data (pd.DataFrame): Dataset
    """
    data["Outlet_Type"] = data["Outlet_Location_Type"].str[-1]
    data["Outlet_Type"][data["Outlet_Type"] == "e"] = 0
    data["Outlet_Type"] = int(data["Outlet_Type"])

def normalize(data, column, clip = False):
    """Normalizes numerical values using min-max normalization

    Args:
        data (pd.DataFrame): Dataset
        column (str): Name of numerical column
        clip (bool, optional): Whether to clip out of range values. Defaults to False.

    Returns:
        np.Ndarray: Normalized array
    """    
    scaler = MinMaxScaler((MIN_MAX[column]["min"],MIN_MAX[column]["max"]),clip=clip)
    normalized_col = scaler.fit_transform(data[column])
    return normalized_col

def one_hot_encode(data, column):
    """Converts labels into one hot encoded data

    Args:
        data (pd.DataFrame): Dataset
        column (str): Name of the categorical column

    Returns:
        np.Ndarray: Encoded data
    """    
    encoder = OneHotEncoder(CLASSES[column])
    encoded_col = encoder.fit_transform(data[column])
    return encoded_col


def data_pipeline(path):
    # read from csv
    data = read_data(path)
    #1 Split Item_Identifier into 3 columns to reduce dimensionality
    split_identifier(data)
    #3 merge categories of Item_Fat_Content
    map_fat(data)
    #9 Replaces outlet size with ordered integers
    set_outlet_size(data)
    #10 Replace tier level text with integer
    extract_tier_level(data)
    #11 Replace Outlet_Type values with ordered integers
    extract_supermarket_level(data)

    processed_data = None
    for column in CATEGORICAL_COLS:
        #5 Directly handle preprocessing of Item_Type.
        #7 Directly handle preprocessing of Outlet_Identifier.
        encoded_col = one_hot_encode(data,column)
        if processed_data is None:
            processed_data = encoded_col
        else:
            processed_data = np.concat([processed_data, encoded_col])
    for column in NORMALIZE_COLS:
        #4 Directly handle preprocessing of Item_Visibility.
        #6 Directly handle preprocessing of Item_MRP.
        #8 Directly handle preprocessing of  Outlet_Establishment_Year. 
        normalized_col = normalize(data, column=column, clip=True)
        processed_data = np.concat([processed_data, normalized_col])
    
    X = processed_data
    y = normalize(data, "Item_Outlet_Sales")
    return X, y
