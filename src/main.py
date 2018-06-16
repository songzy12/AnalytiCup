from util import load_train, load_test, load_unlabel_train, dump_submission
from feature import get_feature
from model import train_model

english_train_path = '../input/cikm_english_train_20180516.txt'
spanish_train_path = '../input/cikm_spanish_train_20180516.txt'
unlabel_spanish_train_path = '../input/cikm_unlabel_spanish_train_20180516.txt'
test_path = '../input/cikm_test_a_20180516.txt'
en_vec_path = '../input/wiki.en.vec'
es_vec_path = '../input/wiki.es.vec'

if __name__ == '__main__':
    english_train_x, english_train_y = load_train(english_train_path)
    spanish_train_x, spanish_train_y = load_train(spanish_train_path)
    unlabel_spanish_train_path = load_unlabel_train(unlabel_spanish_train_path)
    spanish_test_x = load_test(test_path)
    english_train_feature = get_feature(english_train_x)
    test_feature = get_feature(spanish_test_x)
    model = train_model(english_train_feature, english_train_y)
    result = model.predict(test_feature)
    dump_submission(result)
