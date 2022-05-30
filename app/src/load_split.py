import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def split_xy(df, x_col, y_col):
    x_col = ['id']+x_col
    y_col = ['id']+y_col
    return df[x_col], df[y_col]


def get_stratify_col(y, stratify_col):
    if stratify_col is None:
        stratification = None
    else:
        stratification = y[stratify_col]
    
    return stratification


def run_split_data(x, y, stratify_col=None, TEST_SIZE=0.2):
    
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
    joblib.dump(x_train, "output/x_train.pkl")
    joblib.dump(y_train, "output/y_train.pkl")
    joblib.dump(x_valid, "output/x_valid.pkl")
    joblib.dump(y_valid, "output/y_valid.pkl")
    joblib.dump(x_test, "output/x_test.pkl")
    joblib.dump(y_test, "output/y_test.pkl")
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

