import pandas as pd
import numpy as np

from feature import get_feature, get_tokenizer
from model import train_model
from config import english_train_path, spanish_train_path, unlabel_spanish_train_path, test_path

if __name__ == '__main__':
    df_en_train = pd.read_csv(english_train_path, sep='\t', names=['en0', 'es0', 'en1', 'es1', 'label'])
    
    df_es_train = pd.read_csv(spanish_train_path, sep='\t', names=['es0', 'en0', 'es1', 'en1', 'label'])
    df_es2en = pd.read_csv(unlabel_spanish_train_path, sep='\t', names=['es', 'en'])
    df_test = pd.read_csv(test_path, sep='\t', names=['es0', 'es1'])

    tokenizer = get_tokenizer([df_es_train['es0'], df_es_train['es1'], df_test['es0'], df_test['es1']])    

    df_es_train = get_feature(df_es_train, tokenizer)

    predictors = ['word2vec_dot'] + ['word2vec_minkowski_'+str(i) for i in range(1,3)] + \
                 ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'] + \
                 ['jaccard']
                 
    best_model,best_iteration = train_model(df_es_train, predictors)
    
    feature_test = get_feature(df_test, tokenizer)   
    sub = pd.DataFrame()
    sub['result'] = best_model.predict(feature_test[predictors],num_iteration=best_iteration)
    
    sub.to_csv('../output/submission.txt',index=False,header=False,float_format='%.9f')
    print('done.')