
# coding: utf-8

# In[1]:


get_ipython().system('which python')


# In[2]:


get_ipython().system(' conda list tensorflow-gpu ')


# In[3]:


# IF IT DOES NOT WORK, MAY NEED TO RESTART COMPUTER

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0

# confirm PyTorch sees the GPU
from torch import cuda
assert cuda.is_available()  # GIT TEST
assert cuda.device_count() > 0
print(cuda.get_device_name(cuda.current_device()))


# # Reload data

# In[7]:


import numpy as np
import h5py
with h5py.File('data/padded_data.h5','r') as h5f:
    data = h5f['dataset_1'][:]
with h5py.File('data/labels.h5','r') as h5f:
    labels = h5f['dataset_1'][:]


# In[6]:


data


# In[8]:


# split the data into a training set and a validation set
VALIDATION_SPLIT = 0.2

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# In[9]:


get_ipython().run_cell_magic('time', '', "# This is actually super fast\n# first, build index mapping words in the embeddings set\n# to their embedding vector\nimport os \nBASE_DIR = '/home/mritter/code/twitter_nlp/newsgroups_data/'\nGLOVE_DIR = os.path.join(BASE_DIR, 'glove')\n\nprint('Indexing word vectors.')\n\nembeddings_index = {}\nwith open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:\n    for line in f:\n        values = line.split()\n        word = values[0]\n        coefs = np.asarray(values[1:], dtype='float32')\n        embeddings_index[word] = coefs\n\nprint('Found %s word vectors.' % len(embeddings_index))")


# In[11]:


get_ipython().run_cell_magic('time', '', '# prepare embedding matrix\nfrom keras.preprocessing.text import Tokenizer\n\nnum_distinct_words = len(tokenizer.word_index) + 1  # For <UNKNOWN> \nEMBEDDING_DIM = 100  # Dimensions to represent each token\n\nembedding_matrix = np.zeros((num_distinct_words, EMBEDDING_DIM))\nfor word, i in word_index.items():\n    if i > num_distinct_words:\n        continue\n    embedding_vector = embeddings_index.get(word)\n    if embedding_vector is not None:\n        # words not found in embedding index will be all-zeros.\n        embedding_matrix[i] = embedding_vector')


# In[14]:


import h5py
import numpy as np

with h5py.File('data/whole_data.h5', 'r') as h5f:
    embedding_matrix = h5f['embedding_matrix'][:]
    xtrain = h5f['x_train'][:]
    ytrain = h5f['y_train'][:]
    x_val = h5f['x_val'][:]
    y_val = h5f['y_val'][:]


# In[16]:


embedding_matrix.shape


# In[18]:


xtrain.shape


# In[36]:


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.initializers import Constant

num_distinct_words, EMBEDDING_DIM = embedding_matrix.shape
embedding_layer = Embedding(num_distinct_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=xtrain.shape[1],
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(xtrain.shape[1],), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
# model.add(Dense(1, activation='sigmoid'))
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# In[37]:


model.summary()


# In[30]:


# import keras.backend as K
# K.clear_session() 


# In[38]:


# Create a TensorBoard instance with the path to the logs directory
from time import time
from keras.callbacks import TensorBoard as tb
from datetime import datetime
t = datetime.now()
tensorboard = tb(log_dir='tensorboard_logs/{:%Y-%m-%d-%H-%M}'.format(t))

model.fit(xtrain, ytrain,
          batch_size=64, #128,
          epochs=10,
          validation_data=(x_val, y_val),
          callbacks=[tensorboard])

