import numpy as np
import pandas as pd
import time
import joblib
import lightgbm as lgb

from sklearn.metrics import classification_report, roc_auc_score, f1_score, plot_roc_curve, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight


def model_logreg(class_weight = None):
    """
    Function for initiating Logistic Regression Model
    """
    param_dist = {'C' : [0.25, 0.5, 1]}
    base_model = LogisticRegression(random_state=42, solver='liblinear', class_weight=class_weight)
    
    return param_dist, base_model

def model_rf(class_weight = None):
    """
    Function for initiating Random Forest Model
    """
    param_dist = {'n_estimators' : [25, 50, 100]}
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weight)
    
    return param_dist, base_model

def model_lgb(class_weight = None):
    """
    Function for initiating LightGBM Model
    """
    param_dist = {'n_estimators' : [25, 50, 100], 'boosting_type':['gbdt', 'goss']}
    base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight=class_weight)
    
    return param_dist, base_model

def random_search_cv(model, param, scoring, n_iter, x, y, verbosity=0):
    """
    Just a function to run the hyperparameter search
    """
    random_fit = RandomizedSearchCV(estimator = model, 
                                    param_distributions = param, 
                                    scoring = scoring, 
                                    n_iter = n_iter, 
                                    cv = 5, 
                                    random_state = 42, 
                                    verbose = verbosity)
    random_fit.fit(x, y)
    return random_fit

def calibrate_classifier(model, x_valid, y_valid):
    model_calibrated = CalibratedClassifierCV(model, cv='prefit')
    model_calibrated.fit(x_valid, y_valid)
    
    return model_calibrated

def tune_threshold(model, x_valid, y_valid, scorer):
    """
    Function for threshold adjustment
    
    Args:
        - model(callable): Sklearn model
        - x_valid(DataFrame):
        - y_valid(DataFrame):
        - scorer(callable): Sklearn scorer function, for example: f1_score
    """
    thresholds = np.linspace(0,1,101)
    proba = model.predict_proba(x_valid)[:, 1]
    proba = pd.DataFrame(proba)
    proba.columns = ['probability']
    score = []
    for threshold_value in thresholds:
        proba['prediction'] = np.where( proba['probability'] > threshold_value, 1, 0)
        metric_score = scorer(proba['prediction'], y_valid, average='macro')
        score.append(metric_score)
    metric_score = pd.DataFrame([thresholds,score]).T
    metric_score.columns = ['threshold','metric_score']
    best_score = (metric_score['metric_score'] == metric_score['metric_score'].max())
    best_threshold = metric_score[best_score]['threshold']
    
    return metric_score["metric_score"].max(), best_threshold.values[0]

def select_model(train_log_dict):
    max_score = max(train_log_dict['model_score'])
    max_index = train_log_dict['model_score'].index(max_score)
    best_model = train_log_dict['model_fit'][max_index]
    best_report = train_log_dict['model_report'][max_index]
    best_threshold = train_log_dict['threshold'][max_index]
    name = train_log_dict['model_name'][max_index]

    return best_model, best_report, best_threshold, name

def classif_report(model_obj, x_test, y_test, best_threshold=None, calc_auc=True):
    code2rel = {'0': 'Non-Toxic', '1': 'Toxic'}
    
    if best_threshold is None:
        pred = model_obj.predict(x_test)
    else:
        proba = model_obj.predict_proba(x_test)[:, 1]
        pred = np.where(proba > best_threshold, 1, 0)

    res = classification_report(
        y_test, pred, output_dict=True, zero_division=0)
    res = pd.DataFrame(res).rename(columns=code2rel).T

    if calc_auc:
        proba = model_obj.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, proba)

        print(
            f"AUC score: {auc_score}, F1-Macro: {res['f1-score']['macro avg']}")
    return pred, res

def fit(x_train, y_train, model, model_param, scoring='f1', n_iter=3, verbosity=3):
    """
    Fit model
    
    Args:
        - model(callable): sklearn model
        - model_param(dict): sklearn's RandomizedSearchCV params_distribution
    
    Return:
        - model_fitted(callable): model with optimum hyperparams
    """
    model_fitted = random_search_cv(model, model_param, 
                                    scoring, 
                                    n_iter, 
                                    x_train, y_train, 
                                    verbosity)
    print(
        f'Model: {model_fitted.best_estimator_}, {scoring}: {model_fitted.best_score_}')
    
    return model_fitted

def validate(x_valid, y_valid, model_fitted, tune = True):
    """
    Validate model

    Args:
        - x_valid(DataFrame): Validation independent variables
        - y_valid(DataFrame): Validation Dependent variables
        - model_fitted(callable): Sklearn / imblearn fitted model
    """
    code2rel = {'0': 'Non-Toxic', '1': 'Toxic'}

    # Calibrate Classifier
    model_calibrated = CalibratedClassifierCV(base_estimator=model_fitted,
                                              cv="prefit")
    model_calibrated.fit(x_valid, y_valid)
    
    if tune:
        metric_score, best_threshold = tune_threshold(model_calibrated,
                                                      x_valid,
                                                      y_valid,
                                                      f1_score)
        
        print(f'Best threshold is: {best_threshold}, with score: {metric_score}')
        pred_model, report_model = classif_report(model_calibrated,
                                                  x_valid,
                                                  y_valid,
                                                  best_threshold,
                                                  True)
    else:
        # Report default
        best_threshold = None
        pred_model, report_model = classif_report(
            model_calibrated, x_valid, y_valid, True)

    return report_model, model_calibrated, best_threshold

def main(x_train, y_train, x_valid, y_valid, params):
    
    x_train = x_train.drop(columns='id')
    y_train = y_train.drop(columns='id')
    x_valid = x_valid.drop(columns='id')
    y_valid = y_valid.drop(columns='id')
    
    y_train = y_train.values.ravel()
    y_valid = y_valid.values.ravel()

    # Add class weight
    if params['use_weight']:
        class_weight = compute_class_weight(class_weight = 'balanced', 
                                            classes = np.unique(y_train), 
                                            y = y_train)
        class_weights = dict(zip(np.unique(y_train), class_weight))
    else:
        class_weights = None
    
    # Initiate models
    logreg = model_logreg
    rf = model_rf
    lgb = model_lgb
    
    # Initiate logs
    train_log_dict = {'model': [logreg, rf, lgb],
                      'model_name': [],
                      'model_fit': [],
                      'model_report': [],
                      'model_score': [],
                      'threshold': [],
                      'fit_time': []}


    # Try Each models
    for model in train_log_dict['model']:
        param_model, base_model = model(class_weights)
        train_log_dict['model_name'].append(base_model.__class__.__name__)
        print(f'Fitting {base_model.__class__.__name__}')

        # Train
        t0 = time.time()
        scoring = make_scorer(f1_score,average='macro')
        fitted_model = fit(
            x_train, y_train, base_model, param_model, 
            scoring=scoring, verbosity=params['verbosity'])
        elapsed_time = time.time() - t0
        print(f'elapsed time: {elapsed_time} s \n')
        train_log_dict['fit_time'].append(elapsed_time)

        # Validate
        report, calibrated_model, best_threshold = validate(
            x_valid, y_valid, fitted_model)
        train_log_dict['model_fit'].append(calibrated_model)
        train_log_dict['threshold'].append(best_threshold)
        train_log_dict['model_report'].append(report)
        train_log_dict['model_score'].append(report['f1-score']['macro avg'])

    best_model, best_report, best_threshold, name = select_model(
        train_log_dict)
    print(
        f"Model: {name}, Score: {best_report['f1-score']['macro avg']}")
    joblib.dump(best_model, 'output/mantab_model.pkl')
    joblib.dump(best_threshold, 'output/threshold.pkl')
    joblib.dump(train_log_dict, 'output/train_log.pkl')
    print(f'\n {best_report}')
    
    return best_model


