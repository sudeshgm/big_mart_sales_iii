CLASSES = {
    "Item_Identifier_1":[],
    "Item_Identifier_2":[],
    "Item_Identifier_3":[],
    "Outlet_Identifier": []
}
CATEGORICAL_COLS = ["Item_Identifier_1","Item_Identifier_2", "Item_Identifier_3",
                    "Outlet_Identifier"]
NORMALIZE_COLS = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]

FAT_MAP = {
           'Low Fat':0,
           'Regular':1,
           'low fat':0,
           'LF'     :0,
           'reg'   :1
          }

MIN_MAX = {
    "Item_Weight" : {"min":0,"max":25},
    "Item_Visibility" : {"min":0,"max":0.2},
    "Item_MRP": {"min":30,"max":270},
    "Outlet_Establishment_Year": {"min":1985,"max":2009}, 
    "Item_Outlet_Sales": {"min":0,"max":14000}
}