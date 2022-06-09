import re
from tqdm import tqdm
import joblib
from string import punctuation

import nltk
from nltk.corpus import stopwords

tqdm.pandas()

import yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"

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

def load_split_data(params):
    """
    Loader for splitted data.
    
    Args:
    - params(dict): preprocessing params.
    
    Returns:
    - x_train(DataFrame): inputs of train set.
    - x_valid(DataFrame): inputs of valid set.
    - x_test(DataFrame): inputs of test set.
    """

    x_train = joblib.load(params["out_path"]+"x_train.pkl")
    x_valid = joblib.load(params["out_path"]+"x_valid.pkl")
    x_test = joblib.load(params["out_path"]+"x_test.pkl")

    return x_train, x_valid, x_test

def lowercase_char(df_in, do=True):
    """
    Function for lowercasing strings
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].str.lower()
    return df

def phrase_decontraction(phrase):
    """
    Function to decontract phrases
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def decontract(df_in, do=True):
    """
    Main function for decontracting phrases
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(phrase_decontraction)
    return df

def remove_numbers(df_in, do=True):
    """
    Function for removing numbers from text
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: ''.join(string for string in x if not string.isdigit()))
    return df

def remove_punc(df_in, do=True):
    """
    Function for removing punctuation in text
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].str.replace(f'[{punctuation}]', ' ', regex=True )
    return df

def remove_whitespace(df_in, do=True):
    """
    Function for removing whitespace in text
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: " ".join(x.split()))
    return df

def remove_stop(df_in, eng_stopwords, do=True):
    """
    Function for removing stopwords in text
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in eng_stopwords]))
    return df

def preprocess(df_in, params):
    """
    A function to execute the preprocessing steps.
    
    Args:
    - df_in(DataFrame): Input dataframe
    - params(dict): preprocessing parameters
    
    Return:
    - df(DataFrame): preprocessed data
    """
    eng_stopwords = stopwords.words('english')
    df = df_in.copy()
    df = lowercase_char(df, params['lowercase'])
    df = decontract(df, params['decontract'])
    df = remove_numbers(df, params['remove_num'])
    df = remove_punc(df, params['remove_punc'])
    df = remove_whitespace(df, params['remove_space'])
    df = remove_stop(df, eng_stopwords, params['remove_stop'])
    return df

def main_prep(x_train,x_valid,x_test, params):
    x_list = [x_train,x_valid,x_test]

    x_preprocessed = []
    for x in tqdm(x_list):
        temp = preprocess(x, params)
        x_preprocessed.append(temp)

    name = ['train','valid','test']
    for i,x in tqdm(enumerate(x_preprocessed)):
        joblib.dump(x, f"{params['out_path']}x_{name[i]}_preprocessed.pkl")
    
    return x_preprocessed

if __name__ == "__main__":
    params_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)
    x_train, x_valid, x_test = load_split_data(params_preprocess)
    x_preprocessed_list = main_prep(x_train, x_valid, x_test, params_preprocess)