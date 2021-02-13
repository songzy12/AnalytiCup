import string

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors

from fuzzywuzzy import fuzz
import editdistance

from common import es_5w1h_list, en_5w1h_list, es_stop_list, en_stop_list, es_vec_path, embed_size, en


def preprocess(df_train, df_test, en=False):
    if en:
        for df in [df_train]:
            for col in ['en0', 'en1']:
                for token in "¡¿" + string.punctuation:
                    df[col] = df[col].apply(
                        lambda row: row.replace(token, ' ' + token + ' ').lower())

        for df in [df_train, df_test]:
            for col in ['en0', 'en1']:
                df['seq_' + col] = df.apply(lambda row: list(filter(lambda token: token not in list(
                    "¡¿" + string.punctuation) + stopwords.words('english'), word_tokenize(row[col]))), axis=1)
        return

    for df in [df_train, df_test]:
        for col in ['es0', 'es1']:
            for token in "¡¿" + string.punctuation:
                df[col] = df[col].apply(
                    lambda row: row.replace(token, ' ' + token + ' ').lower())

    for df in [df_train, df_test]:
        for col in ['es0', 'es1']:
            df['seq_' + col] = df.apply(lambda row: list(filter(lambda token: token not in list(
                "¡¿" + string.punctuation) + stopwords.words('spanish'), word_tokenize(row[col]))), axis=1)


def get_feature_cnt(df, en):

    for token in (es_stop_list if not en else en_stop_list):
        df['stop_' + token] = df.apply(
            lambda row: token in (row['es0'] if not en else row['en0']) and token in (row['es1'] if not en else row['en1']), axis=1)
    for token in (es_5w1h_list if not en else en_5w1h_list):
        df['5w1h_' + token] = df.apply(
            lambda row: token in (row['es0'] if not en else row['en0']) and token in (row['es1'] if not en else row['en1']), axis=1)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


w2v_model = KeyedVectors.load_word2vec_format(
    '../input/wiki.es.vec' if not en else '../input/wiki.en.vec')
with open(es_vec_path, encoding="utf8") as f:
    f.readline()
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in f)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()


def get_feature_embedding(df, en):
    for col in (['es0', 'es1'] if not en else ['en0', 'en1']):
        df['word2vec_' + col] = df.apply(lambda row: (sum([embeddings_index.get(x, np.random.normal(emb_mean, emb_std, (embed_size))) for x in row[
            'seq_' + col]]) / len(row['seq_' + col])) if len(row['seq_' + col]) else np.random.normal(emb_mean, emb_std, (embed_size)), axis=1)

    if not en:
        df['dot'] = df.apply(lambda row: row['word2vec_es0'].dot(row['word2vec_es1']) / (
            row['word2vec_es0'].dot(row['word2vec_es0'])**0.5 * row['word2vec_es1'].dot(row['word2vec_es1'])**0.5), axis=1)

        df['wmd'] = df.apply(lambda row: w2v_model.wmdistance(
            row['seq_es0'], row['seq_es1']), axis=1)
    else:
        df['dot'] = df.apply(lambda row: row['word2vec_en0'].dot(row['word2vec_en1']) / (
            row['word2vec_en0'].dot(row['word2vec_en0'])**0.5 * row['word2vec_en1'].dot(row['word2vec_en1'])**0.5), axis=1)

        df['wmd'] = df.apply(lambda row: w2v_model.wmdistance(
            row['seq_en0'], row['seq_en1']), axis=1)

    def minkowski_distance(x, y, p_value):
        # pass the p_root function to calculate
        # all the value of vector parallely
        def p_root(value, root):
            root_value = 1 / float(root)
            return round(value ** root_value, 3)

        return (p_root(sum(pow(abs(a - b), p_value)
                           for a, b in zip(x, y)), p_value))

    for i in range(1, 3):
        if not en:
            df['minkowski_' + str(i)] = df.apply(
                lambda row: minkowski_distance(row['word2vec_es1'], row['word2vec_es0'], i), axis=1)
        else:
            df['minkowski_' + str(i)] = df.apply(
                lambda row: minkowski_distance(row['word2vec_en1'], row['word2vec_en0'], i), axis=1)

    return df


def get_feature_distance(df, en):
    if not en:
        df['edit_distance'] = df.apply(
            lambda row: editdistance.eval(row['seq_es0'], row['seq_es1']), axis=1)
    else:
        df['edit_distance'] = df.apply(
            lambda row: editdistance.eval(row['seq_en0'], row['seq_en1']), axis=1)
    return df


def get_feature_set(df, en):

    df['ratio'] = df.apply(
        lambda row: fuzz.ratio(row['seq_es0' if not en else 'seq_en0'], row['seq_es1' if not en else 'seq_en1']), axis=1)
    df['partial_ratio'] = df.apply(
        lambda row: fuzz.partial_ratio(row['seq_es0' if not en else 'seq_en0'], row['seq_es1' if not en else 'seq_en1']), axis=1)
    df['token_sort_ratio'] = df.apply(
        lambda row: fuzz.token_sort_ratio(row['seq_es0' if not en else 'seq_en0'], row['seq_es1' if not en else 'seq_en1']), axis=1)
    df['token_set_ratio'] = df.apply(
        lambda row: fuzz.token_set_ratio(row['seq_es0' if not en else 'seq_en0'], row['seq_es1' if not en else 'seq_en1']), axis=1)

    def jaccard(a, b):
        a = set(a)
        b = set(b)
        c = a.intersection(b)
        if len(a) + len(b) - len(c) == 0:
            return 0
        return float(len(c)) / (len(a) + len(b) - len(c))
    df['jaccard'] = df.apply(
        lambda row: jaccard(row['seq_es0' if not en else 'seq_en0'], row['seq_es1' if not en else 'seq_en1']), axis=1)
    return df


def get_feature(df, en):
    get_feature_cnt(df, en)
    get_feature_embedding(df, en)
    get_feature_distance(df, en)
    get_feature_set(df, en)
