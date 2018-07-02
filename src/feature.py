import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

from config import en_vec_path, es_vec_path, embed_size, max_features, maxlen

def get_tokenizer(texts):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(pd.concat(texts)))
    return tokenizer

def get_vec_feature(df, tokenizer):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    with open(es_vec_path, encoding="utf8") as f:
        f.readline()
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in f)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
    
    df['seq_es0'] = tokenizer.texts_to_sequences(df['es0'])
    df['seq_es1'] = tokenizer.texts_to_sequences(df['es1'])


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

    df['word2vec_es0'] = [sum([embedding_matrix[x] if (x < len(embedding_matrix)) else emb_mean for x in data ])/len(data) for data in df['seq_es0']]
    df['word2vec_es1'] = [sum([embedding_matrix[x] if (x < len(embedding_matrix)) else emb_mean for x in data ])/len(data) for data in df['seq_es1']]
    df['word2vec_dot'] = df.apply(lambda row: row['word2vec_es0'].dot(row['word2vec_es1'])/(row['word2vec_es0'].dot(row['word2vec_es0'])**0.5 * row['word2vec_es1'].dot(row['word2vec_es1'])**0.5), axis=1)
    return df


def get_feature(df, tokenizer):
    df = get_vec_feature(df, tokenizer)
    return df
