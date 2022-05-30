import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import joblib

def lowercase_char(df_in, do=True):
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].str.lower()
    return df

def phrase_decontraction(phrase):
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
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(phrase_decontraction)
    return df

def remove_numbers(df_in, do=True):
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: ''.join(string for string in x if not string.isdigit()))
    return df

def remove_punc(df_in, do=True):
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].str.replace(f'[{punctuation}]', ' ', regex=True )
    return df

def remove_whitespace(df_in, do=True):
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: " ".join(x.split()))
    return df

def remove_stop(df_in, eng_stopwords, do=True):
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['comment_text'] = df['comment_text'].apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in eng_stopwords]))
    return df

def preprocess(df_in, params):
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
        joblib.dump(x, f"output/x_{name[i]}_preprocessed.pkl")
    
    return x_preprocessed
