import pandas as pd



def load_data(file_path: str):
    return pd.read_csv(file_path) 



def preprocess_data(df: pd.DataFrame):
    
    df = df.drop_duplicates().copy()


    # split into x and y
    x = df.drop("Class",axis =1)
    y = df['Class']
    

    return x, y