import joblib
import pandas as pd
from preprocessing import preprocess
from feature_engineering import vectorize_tfidf

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

tfidf_vect = joblib.load("../output/vectorizer.pkl")
model = joblib.load('../model/mantab_model.pkl')
threshold = joblib.load('../model/threshold.pkl')

def df_constructor(text, id=0):
    df = pd.DataFrame(data={'id':[id], 'comment_text':[text]})
    return df

def main_predict(text, tfidf_vectorizer, model, threshold, param_preprocess, param_vec, id=0):
    df = df_constructor(text, id)
    df_preprocessed = preprocess(df, param_preprocess)
    df_vect, _ = vectorize_tfidf(df_preprocessed, param_vec, tfidf_vectorizer)
    
    code2rel = {0: 'Non-Toxic', 1: 'Toxic'}
    df_vect = df_vect.drop(columns='id')
    proba = model.predict_proba(df_vect)[:, 1]
    predict = 1 if proba > threshold else 0
    
    return code2rel[predict], proba

param_vec = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
params_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)

if __name__ == "__main__":
    while(1):
        print("Masukkan text yang ingin anda klasifikasikan:")
        text = input()
        predict, proba = main_predict(text, tfidf_vect, model, threshold, params_preprocess, param_vec)
        print("Hasil prediksi\t:", predict)
        print("Probabilitas\t:", proba[0], "\n\n")
