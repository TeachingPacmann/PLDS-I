import pandas as pd
from tqdm import tqdm
import joblib

tqdm.pandas()

import preprocessing
import feature_engineering
from predict import main_predict
from fastapi import FastAPI, Form, Header, HTTPException
from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

tfidf_vect = joblib.load("../output/vectorizer.pkl")
model = joblib.load('../model/mantab_model.pkl')
threshold = joblib.load('../model/threshold.pkl')

app = FastAPI()

params_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)
param_vec = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)

def res_constructor(predict, proba):
    res = {'result' : predict, 'proba' : proba[0], 'message' : ""}
    return res

@app.post("/predict/")
def predict_api(text = Form(), x_token = Header(...)):
    if x_token != "passowrd_plds":
        raise HTTPException(status_code=401, detail='invalid X-Token header')

    try:
        predict, proba = main_predict(text, tfidf_vect, model, threshold, params_preprocess, param_vec, id)
        res = res_constructor(predict, proba)
        return res
    
    except Exception as e:
        return {'result' : "", 'proba' : "", 'message' : str(e)}
