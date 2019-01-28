# Inputs
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'
training_file = 'indexed.h5'

# Knobs
pass  # The whole thing, to some extent

# Outputs
compiled_model = 'compiled_model.h5'

import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'


with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "r") as f:
    embedding_matrix = f['weights'][()]
    embedding_dim = f['metadata/embedding_dim'][0]
    num_distinct_words = f['metadata/max_token'][0]
    sequence_length = f['metadata/sequence_length'][0]

from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.initializers import Constant
from keras.models import Model


sequence_input = Input(shape=(sequence_length,), dtype='int32')

embedded_sequences = Embedding(num_distinct_words,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=sequence_length,
                            trainable=False)(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=32, activation='relu')(x)
preds = Dense(units=2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())  # Dial

model.save(DATADIR+compiled_model)
# serialize model to YAML
# From https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#model_yaml = model.to_yaml()
#with open(DATADIR+compiled_model, "w") as yaml_file:
#    yaml_file.write(model_yaml)

# serialize weights to HDF5
#model.save_weights(DATADIR+compiled_weights)
#print("Saved model to disk")
