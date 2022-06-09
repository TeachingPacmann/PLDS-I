import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split

tqdm.pandas()

import yaml

LOAD_SPLIT_CONFIG_PATH = "../config/load_split_config.yaml"

def read_yaml(yaml_path):
    """
    Loader for yaml file.
    
    Args:
    - yaml_path(string): Path to yaml file.
    
    Returns:
    - params(dictionary): Dict ver of yaml file.
    """
    
    with open(yaml_path, "r") as stream:
        params = yaml.safe_load(stream)
    
    return params 

def split_xy(df, x_col, y_col):
    """
    Splitting x and y variables.
    
    Args:
    - df(DataFrame): initial input dataframe
    - x_col(list): List of x variable columns
    - y_col(list): List of y variable columns
    
    Returns:
    - df[x_col](DataFrame): Dataframe contains x columns and id
    - df[y_col](DataFrame): Dataframe contains y columns and id
    """
    x_col = ['id']+x_col
    y_col = ['id']+y_col
    return df[x_col], df[y_col]


def get_stratify_col(y, stratify_col):
    """
    Splitting x and y variables.
    
    Args:
    - y(DataFrame): DataFrame contains target variables and id
    - stratify_col(str): column name of the reference column.
    
    Returns:
    - stratification: Dataframe contains column that will be used as stratification reference
    """
    if stratify_col is None:
        stratification = None
    else:
        stratification = y[stratify_col]
    
    return stratification


def run_split_data(x, y, stratify_col=None, TEST_SIZE=0.2):
    """
    Splitting x and y variables.
    
    Args:
    - y(DataFrame): DataFrame contains predictor variables and id
    - y(DataFrame): DataFrame contains target variables and id
    - stratify_col(str): column name of the reference column.
    - TEST_SIZE(float): Size of the test and validation dataset size.
    
    Returns:
    - x_blabla(DataFrame): X variables for train/valid/test dataset
    - y_blabla(DataFrame): Y variables for train/valid/test dataset
    """
    strat_train = get_stratify_col(y, stratify_col)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                       stratify = strat_train,
                                       test_size= TEST_SIZE*2,
                                       random_state= 42)
    
    strat_test = get_stratify_col(y_test, stratify_col)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test,
                                       stratify = strat_test,
                                       test_size= 0.5,
                                       random_state= 42)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def main_load(params):
    df = pd.read_csv(params["file_loc"])
    x_all, y_all = split_xy(df, [params['x_col']], [params['y_col']])
    x_train, y_train,x_valid, y_valid,x_test, y_test = run_split_data(x_all, y_all, 
                                                                      params['stratify'], 
                                                                      params['test_size'])
    joblib.dump(x_train, params["out_path"]+"x_train.pkl")
    joblib.dump(y_train, params["out_path"]+"y_train.pkl")
    joblib.dump(x_valid, params["out_path"]+"x_valid.pkl")
    joblib.dump(y_valid, params["out_path"]+"y_valid.pkl")
    joblib.dump(x_test, params["out_path"]+"x_test.pkl")
    joblib.dump(y_test, params["out_path"]+"y_test.pkl")
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == "__main__":
    params = read_yaml(LOAD_SPLIT_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid, x_test, y_test = main_load(params)