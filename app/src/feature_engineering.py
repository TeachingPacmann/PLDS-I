import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_tfidf(df_in, params, vectorizer=None):
    df = df_in.copy()
    if vectorizer is None:  # fit to train data
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            min_df = params['min_df']
        )
        vectorized = vectorizer.fit_transform(df['comment_text'])
        joblib.dump(vectorizer, f"output/{params['vectorizer_file']}.pkl")
    else:
        vectorized = vectorizer.transform(df['comment_text'])
    
    vectorized_df = pd.DataFrame(vectorized.toarray(), 
                                 columns=vectorizer.get_feature_names(), 
                                 index = df.index)
    df_non_sentence = df.drop(['comment_text'],axis=1)
    df_final = pd.concat([vectorized_df, df_non_sentence],axis=1)
    return df_final, vectorizer

def main_feat(x_preprocessed_list, params):
    x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed = x_preprocessed_list
    df_train_vect, vectorizer = vectorize_tfidf(x_train_preprocessed, params)
    df_valid_vect, _ = vectorize_tfidf(x_valid_preprocessed, params, vectorizer)
    df_test_vect, _ = vectorize_tfidf(x_test_preprocessed, params, vectorizer)
    joblib.dump(df_train_vect, f"output/x_train_vect.pkl")
    joblib.dump(df_valid_vect, f"output/x_valid_vect.pkl")
    joblib.dump(df_test_vect, f"output/x_test_vect.pkl")
    
    return df_train_vect, df_valid_vect, df_test_vect

