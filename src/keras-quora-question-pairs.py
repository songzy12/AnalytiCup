from __future__ import print_function
import numpy as np
import pandas as pd
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split

from common import english_train_path, spanish_train_path, unlabel_spanish_train_path, test_path
from common import en_vec_path, es_vec_path, embed_size, max_features, maxlen

# Initialize global variables

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25
DROPOUT = 0.1
BATCH_SIZE = 32
OPTIMIZER = 'adam'

   
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

if __name__ == '__main__':
    df_en_train = pd.read_csv(english_train_path, sep='\t', names=['en0', 'es0', 'en1', 'es1', 'label'])
    
    df_es_train = pd.read_csv(spanish_train_path, sep='\t', names=['es0', 'en0', 'es1', 'en1', 'label'])
    df_es2en = pd.read_csv(unlabel_spanish_train_path, sep='\t', names=['es', 'en'])
    df_test = pd.read_csv(test_path, sep='\t', names=['es0', 'es1'])

    df_train = pd.concat([df_en_train, df_es_train], ignore_index=True)
    question1 = df_train['es0']
    question2 = df_train['es1']
    test1 = df_test['es0']
    test2 = df_test['es1']
    is_duplicate = df_train['label']

    print('Question pairs: %d' % len(question1))

    # Build tokenized word index
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    test1_word_sequences = tokenizer.texts_to_sequences(test1)
    test2_word_sequences = tokenizer.texts_to_sequences(test2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    with open(es_vec_path, encoding="utf8") as f:
        f.readline()
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in f)

        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()


    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test1_data = pad_sequences(test1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test2_data = pad_sequences(test2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)


    # Partition the dataset into train and test sets
    X = np.stack((q1_data, q2_data), axis=1)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
    Q1_train = X_train[:,0]
    Q2_train = X_train[:,1]
    Q1_test = X_test[:,0]
    Q2_test = X_test[:,1]

    # Define the model
    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nb_words + 1, 
                     EMBEDDING_DIM, 
                     weights=[word_embedding_matrix], 
                     input_length=MAX_SEQUENCE_LENGTH, 
                     trainable=False)(question1)
    q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

    q2 = Embedding(nb_words + 1, 
                     EMBEDDING_DIM, 
                     weights=[word_embedding_matrix], 
                     input_length=MAX_SEQUENCE_LENGTH, 
                     trainable=False)(question2)
    q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

    merged = concatenate([q1,q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1,question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    # Train the model, checkpointing weights with best validation accuracy
    print("Starting training at", datetime.datetime.now())
    t0 = time.time()
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]
    history = model.fit([Q1_train, Q2_train],
                        y_train,
                        epochs=NB_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=2,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)
    t1 = time.time()
    print("Training ended at", datetime.datetime.now())
    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

    # Print best validation accuracy and epoch
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))

    # Evaluate the model with best validation accuracy on the test partition
    model.load_weights(MODEL_WEIGHTS_FILE)
    loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
    print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))
    result = model.predict([test1_data, test2_data])

    sub = pd.DataFrame(result)
    sub.to_csv('../output/quora-submission.txt',index=False,header=False,float_format='%.9f')
    print('done.')
