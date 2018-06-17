import pandas

from util import dump_submission
from feature import get_feature
from model import train_model

english_train_path = '../input/cikm_english_train_20180516.txt'
spanish_train_path = '../input/cikm_spanish_train_20180516.txt'
unlabel_spanish_train_path = '../input/cikm_unlabel_spanish_train_20180516.txt'
test_path = '../input/cikm_test_a_20180516.txt'
en_vec_path = '../input/wiki.en.vec'
es_vec_path = '../input/wiki.es.vec'

if __name__ == '__main__':
    df_en_train = pandas.read_csv(english_train_path, sep='\t', names=['en0', 'es0', 'en1', 'es1', 'label'])
    df_es_train = pandas.read_csv(spanish_train_path, sep='\t', names=['es0', 'en0', 'es1', 'en1', 'label'])
    df_es2en = pandas.read_csv(unlabel_spanish_train_path, sep='\t', names=['es', 'en'])
    df_test = pandas.read_csv(test_path, sep='\t', names=['es0', 'es1'])
    feature_train_es = get_feature(df_es_train.get(['es0', 'es1']))
    feature_test_es = get_feature(df_test)
    model = train_model(feature_train_es, df_en_train.get('label'))
    result = model.predict(feature_test_es)
    dump_submission(result)
