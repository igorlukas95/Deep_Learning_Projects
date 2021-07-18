# -*- coding: utf-8 -*-
#%% Importing dataset
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
X_train[0][:10]
#%% Decoding and encoding words
word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(('<pad>', '<sos>', '<unk>')):
    id_to_word[id_] = token
#%%
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])
from tensorflow import tensorflow_data
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
#%%
