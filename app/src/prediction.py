import joblib


import pandas as pd
from .preprocess import preprocess
from .feature_engineering import vectorize_tfidf


# In the previous preprocessing, we work with DataFrame.
# It'll be easier for us to also work with DataFrame in the prediction stage
                      
def df_constructor(text, id=0):
    df = pd.DataFrame(data={'id':[id], 'comment_text':[text]})
    return df

def main_predict(text, tfidf_vectorizer, model, threshold, param_preprocess, param_vec, id=0):
    df = df_constructor(text, id)
    df_preprocessed = preprocess(df, param_preprocess)
    df_vect, _ = vectorize_tfidf(df_preprocessed, param_vec, tfidf_vectorizer)
    
    # code2rel = {0: 'Non-Toxic', 1: 'Toxic'}
    df_vect = df_vect.drop(columns='id')
    proba = model.predict_proba(df_vect)[:, 1]
    predict = True if proba > threshold else False
    return predict, proba[0]