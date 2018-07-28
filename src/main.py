import string

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

# from feature import preprocess, get_feature
from model import train_model, load_model
from common import *

if __name__ == '__main__':
    df_en_train = pd.read_csv(english_train_path, sep='\t', names=[
                              'en0', 'es0', 'en1', 'es1', 'label'])

    df_es_train = pd.read_csv(spanish_train_path, sep='\t', names=[
                              'es0', 'en0', 'es1', 'en1', 'label'])
    df_es2en = pd.read_csv(unlabel_spanish_train_path,
                           sep='\t', names=['es', 'en'])
    df_test = pd.read_csv(test_path, sep='\t', names=['es0', 'es1'])

    # preprocess(df_es_train, df_test)

    # get_feature(df_es_train)
    # df_es_train.to_pickle('../output/df_es_train.pkl')
    df_es_train = pd.read_pickle('../output/df_es_train.pkl')

    print(len(df_es_train))

    predictors = ['dot'] + ['minkowski_' + str(i) for i in range(1, 3)] + ['wmd'] + \
                 ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'] + ['jaccard'] +\
                 ['edit_distance'] + \
                 ['stop_' + token for token in es_stop_list] + \
                 ['5w1h_' + token for token in es_5w1h_list]
    target = 'label'

    best_model, best_iteration = train_model(df_es_train, predictors, target, 1200)
    best_model.save_model('../output/model_es.txt')

    best_model = load_model('../output/model_es.txt')
    # get_feature(df_test)
    # df_test.to_pickle('../output/df_test.pkl')
    df_test = pd.read_pickle('../output/df_test.pkl')

    sub = pd.DataFrame()
    sub['result'] = best_model.predict(df_test[predictors])
    sub.to_csv('../output/submission.txt', index=False,
               header=False, float_format='%.9f')
