from .preprocess import main_prep
from .feature_engineering import main_feat
from .load_split import main_load
from .model import main

params_preprocess = { 'lowercase': True, 
                      'decontract':True, 
                      'remove_num':True, 
                      'remove_punc': True, 
                      'remove_space': True, 
                      'remove_stop': True}

param_vec = {'min_df':0.01, 
             'vectorizer_file': 'vectorizer'}

params = {'file_loc': 'data/comments_data.csv', 
          'x_col':'comment_text', 
          'y_col':'toxic', 
          'stratify': 'toxic', 
          'test_size':0.2}

params_preprocess = { 'lowercase': True, 
                      'decontract':True, 
                      'remove_num':True, 
                      'remove_punc': True, 
                      'remove_space': True, 
                      'remove_stop': True}
param_model={'use_weight':True, 
             'verbosity':2}

param_vec = {'min_df':0.01, 
             'vectorizer_file': 'vectorizer'}


x_train, y_train, x_valid, y_valid, x_test, y_test = main_load(params)
x_preprocessed_list = main_prep(x_train,x_valid,x_test,params_preprocess)
x_train_vect, x_valid_vect, x_test_vect = main_feat(x_preprocessed_list, param_vec)
best_model = main(x_train_vect, y_train, x_valid_vect, y_valid, param_model)