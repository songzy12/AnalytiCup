import pandas as pd
import numpy as np

from util import dump_submission
from feature import get_feature, get_tokenizer
from model import train_model
from config import english_train_path, spanish_train_path, unlabel_spanish_train_path, test_path

if __name__ == '__main__':
    df_en_train = pd.read_csv(english_train_path, sep='\t', names=['en0', 'es0', 'en1', 'es1', 'label'])
    
    df_es_train = pd.read_csv(spanish_train_path, sep='\t', names=['es0', 'en0', 'es1', 'en1', 'label'])
    df_es2en = pd.read_csv(unlabel_spanish_train_path, sep='\t', names=['es', 'en'])
    df_test = pd.read_csv(test_path, sep='\t', names=['es0', 'es1'])

    tokenizer = get_tokenizer([df_es_train['es0'], df_es_train['es1'], df_test['es0'], df_test['es1']])    

    feature_train_es = get_feature(df_es_train, tokenizer)
    model = train_model(feature_train_es, df_es_train.get('label'))
    feature_test_es = get_feature(df_test, tokenizer)
    result = model.predict(feature_test_es)
    dump_submission(result)