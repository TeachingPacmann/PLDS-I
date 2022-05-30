import joblib
# import json 
# from flask import Flask, request, jsonify

import logging.config

# from app.src.feature_engineering_input import construct_df, feature_engineering_predict
# from app.src.modelling import modelling
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.exception import BaseException, validation_exception_handler, starlette_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.src.prediction import main_predict
from app.src.preprocess import main_prep
from app.src.feature_engineering import main_feat
from app.src.load_split import main_load

from app.http_response import ok
# from .model import main


# app = Flask(__name__)

params_preprocess = { 'lowercase': True, 
                      'decontract':True, 
                      'remove_num':True, 
                      'remove_punc': True, 
                      'remove_space': True, 
                      'remove_stop': True}

param_vec = {'min_df':0.01, 
             'vectorizer_file': 'vectorizer'}

tfidf_vect = joblib.load("output/vectorizer.pkl")
model = joblib.load('output/mantab_model.pkl')
threshold = joblib.load('output/threshold.pkl')


app = FastAPI(
    exception_handlers={
        RequestValidationError: validation_exception_handler,
        StarletteHTTPException: starlette_exception_handler,
    },
)

# setup loggers
logging.config.fileConfig('app/logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)

class Item(BaseModel):
    text: str

fake_secret_token = "pacmannpmdata"


@app.get("/")
async def read_root():
    return {"msg": "Hello World"}

# first endpoint
@app.get("/log_now")
def log_now():
    logger.info("logging from the root logger")

    return {"result": "OK"}

@app.post("/predict")
def predict(item: Item, x_token: str = Header(...)):
    if x_token != fake_secret_token:
        raise HTTPException(status_code=401, detail="Invalid X-Token header")

    try:
        data = item.dict()
        predict, proba = main_predict(data['text'], tfidf_vect, model, threshold,params_preprocess, param_vec)
        logger.info("finished predict")
        return ok(data = {
                            "is_toxic" : predict,
                            "proba" : proba
                    }
                )

    except Exception as e:
        message = "There is error in our config!"
        logger.error(e)
        raise BaseException(message=message)