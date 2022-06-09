import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()

import yaml

FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

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

def load_preprocessed_data(params):
    """
    Loader for preprocessed data.
    
    Args:
    - params(dict): preprocessing params.
    
    Returns:
    - list_of_preprocessed(List): list of preprocessed data.
    """

    name = ['train','valid','test']
    list_of_preprocessed = []
    for i in name:
        path = f"{params['out_path']}x_{i}_preprocessed.pkl"
        temp = joblib.load(path)
        list_of_preprocessed.append(temp)

    return list_of_preprocessed

def vectorize_tfidf(df_in, params, vectorizer=None):
    """
    function to execute vectorization using tfidf
    
    Args:
    - df_in(DataFrame): Input data
    - params(dict): Vectorizer parameters
    - vectorizer(callable): tfidf vectorizer, default to None. 
    If None, then the function will create a new tfidf vectorizer  
    """
    df = df_in.copy()
    if vectorizer is None:  # fit to train data
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            min_df = params['min_df']
        )
        vectorized = vectorizer.fit_transform(df['comment_text'])
        joblib.dump(vectorizer, f"../output/{params['vectorizer_file']}.pkl")
    else:
        vectorized = vectorizer.transform(df['comment_text'])
    
    vectorized_df = pd.DataFrame(vectorized.toarray(), 
                                 columns=vectorizer.get_feature_names_out(), 
                                 index = df.index)
    df_non_sentence = df.drop(['comment_text'],axis=1)
    df_final = pd.concat([vectorized_df, df_non_sentence],axis=1)
    return df_final, vectorizer

def main_feat(x_preprocessed_list, params):
    """
    Main function for feature engineering
    """
    x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed = x_preprocessed_list
    df_train_vect, vectorizer = vectorize_tfidf(x_train_preprocessed, params)
    df_valid_vect, _ = vectorize_tfidf(x_valid_preprocessed, params, vectorizer)
    df_test_vect, _ = vectorize_tfidf(x_test_preprocessed, params, vectorizer)
    joblib.dump(df_train_vect, f"{params['out_path']}x_train_vect.pkl")
    joblib.dump(df_valid_vect, f"{params['out_path']}x_valid_vect.pkl")
    joblib.dump(df_test_vect, f"{params['out_path']}x_test_vect.pkl")
    
    return df_train_vect, df_valid_vect, df_test_vect

if __name__ == "__main__":
    param_vec = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
    x_preprocessed_list = load_preprocessed_data(param_vec)
    x_train_vect, x_valid_vect, x_test_vect = main_feat(x_preprocessed_list, param_vec)