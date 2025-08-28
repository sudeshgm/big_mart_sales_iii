import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CLASSES = {
    "Item_Fat_Content":[],
    "Outlet_Size": [],
    "Outlet_Location_Type":[],
    "Outlet_Type":[]
}
CATEGORICAL_COLS = ["Item_Identifier_1","Item_Identifier_2", "Item_Identifier_3",
                    "Item_Fat_Content","Outlet_Identifier", "Outlet_Size",
                    "Outlet_Location_Type","Outlet_Type",]

FAT_MAP = {
           'Low Fat':'L',
           'Regular':'R',
           'low fat':'L',
           'LF'     :'L',
           'reg'   :'R'
          }

def read_data(path):
    return pd.read_csv(path)

def one_hot_encode(data, column,drop=False):
    encoder = OneHotEncoder(CLASSES[column])
    encoded_col = encoder.fit_transform(data[column])
    if drop:
        data.drop(column=[column], inplace=True)
    return encoded_col

def map_fat(data):
    for key in FAT_MAP.keys():
        data["Item_Fat_Content"][data["Item_Fat_Content"] == key] = FAT_MAP[key]

def split_identifier(data, drop=False):
    data["Item_Identifier_1"] = data["Item_Identifier"].str[0:2]
    data["Item_Identifier_2"] = data["Item_Identifier"].str[2:3]
    data["Item_Identifier_3"] = data["Item_Identifier"].str[3:]
    if drop:
        data.drop(column=["Item_Identifier"],inplace=True)

def clip_item_visibility(data, threshold = 0.2):
    data["Item_Visibility"] > threshold = threshold

def data_pipeline(path):
    # read from csv
    data = read_data(path)
    # Split Item_Identifier into 3 columns to reduce dimensionality
    split_identifier(data)
    # merge categories of Item_Fat_Content
    map_fat(data)
    # Clip values of Item_Visibility to nullify effects of outliers
    clip_item_visibility(data)


    processed_data = None
    for column in CATEGORICAL_COLS:
        encoded_col = one_hot_encode(data,column,drop=True)
        if processed_data is None:
            processed_data = encoded_col
        else:
            processed_data = np.concat([processed_data, encoded_col])
