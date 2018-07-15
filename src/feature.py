from math import *

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

from fuzzywuzzy import fuzz

from config import en_vec_path, es_vec_path, embed_size, max_features, maxlen


    
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
with open(es_vec_path, encoding="utf8") as f:
    f.readline()
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in f)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

def get_tokenizer(texts):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(pd.concat(texts)))
    return tokenizer

def get_feature_vec(df, tokenizer):
    # NOTE: dictionary mapping words (str) to their rank/index (int)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    print('max_features:',max_features,'len(word_index):',len(word_index))
    # NOTE: here is how we can randomly initialize unknown word embeddings
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    df['word2vec_es0'] = [sum([embedding_matrix[x] for x in data])/len(data) if data else np.random.normal(emb_mean, emb_std, (embed_size)) for data in df['seq_es0']]
    df['word2vec_es1'] = [sum([embedding_matrix[x] for x in data])/len(data) if data else np.random.normal(emb_mean, emb_std, (embed_size)) for data in df['seq_es1']]

    def dot(row):
        return row['word2vec_es0'].dot(row['word2vec_es1'])/(row['word2vec_es0'].dot(row['word2vec_es0'])**0.5 * row['word2vec_es1'].dot(row['word2vec_es1'])**0.5)
    
    df['word2vec_dot'] = df.apply(dot, axis=1)    

    def p_root(value, root):
         
        root_value = 1 / float(root)
        return round (value ** root_value, 3)
     
    def minkowski_distance(x, y, p_value):
         
        # pass the p_root function to calculate
        # all the value of vector parallely 
        return (p_root(sum(pow(abs(a-b), p_value)
                for a, b in zip(x, y)), p_value))

    for i in range(1, 3):
        df['word2vec_minkowski_'+str(i)] = df.apply(lambda row: minkowski_distance(row['word2vec_es1'], row['word2vec_es0'], i), axis=1)

    return df

def get_feature_edit_distance(df):
    df['ratio'] = df.apply(lambda row: fuzz.ratio(row['seq_es0'], row['seq_es1']), axis=1)
    df['partial_ratio'] = df.apply(lambda row: fuzz.partial_ratio(row['seq_es0'], row['seq_es1']), axis=1)
    df['token_sort_ratio'] = df.apply(lambda row: fuzz.token_sort_ratio(row['seq_es0'], row['seq_es1']), axis=1)
    df['token_set_ratio'] = df.apply(lambda row: fuzz.token_set_ratio(row['seq_es0'], row['seq_es1']), axis=1)
    return df

def get_feature_jaccard(df):
    def jaccard(a, b):
        a = set(a)
        b = set(b)
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))    
    df['jaccard'] =  df.apply(lambda row: jaccard(row['seq_es0'], row['seq_es1']), axis=1)
    return df

def get_feature(df, tokenizer):
    
    df['seq_es0'] = tokenizer.texts_to_sequences(df['es0'])
    df['seq_es1'] = tokenizer.texts_to_sequences(df['es1'])

    df = get_feature_vec(df, tokenizer)
    df = get_feature_edit_distance(df)
    df = get_feature_jaccard(df)
    return df
