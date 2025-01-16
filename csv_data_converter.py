#%% working directory

import os
import pandas as pd

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
# path = "C:\Studia\Market-analysis"
os.chdir(path)

#%%

def load_data_from_file(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        df = pd.read_csv(path)
    return df

def repair_df(df):
    # renaming the first column to date
    df = df.rename(columns={"Price": "date"})
    # dropping first 2 empty rows 
    # df = df.drop([0, 1])
    df = df.iloc[2:]
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"adj close": "adjusted_close"})
    return df

def convert_data(directory_name):
    for file_name in os.listdir(directory_name):
        file_path = os.path.join(directory_name, file_name)
        df = load_data_from_file(file_path)
        df = repair_df(df)
        df.to_csv(file_path, index=False)
        
def main():
    directory_name = "data"
    print("Starting converting data...")
    convert_data(directory_name)
    print("Successfully converted data")
    
if __name__ == "__main__":
    main()