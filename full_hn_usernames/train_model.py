# Input
compiled_model = 'compiled_model.h5'
training_file = 'indexed.h5'
label_file = 'label_and_index.h5'

# Knobs
epochs = 5
batch_size = 64

# Output
trained_model = 'trained_model.yaml'
trained_weights = 'trained_weights.h5'


import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'

from keras.models import load_model

# load YAML and create model
#with open(DATADIR+trained_model, 'r') as yaml_file:
#    loaded_model_yaml = yaml_file.read()
#    loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
#loaded_model.load_weights(DATADIR+trained_weights)
model = load_model(DATADIR+compiled_model)
print("Loaded model from disk")


from keras.callbacks import TensorBoard as tb
from datetime import datetime
t = datetime.now()
tensorboard = tb(log_dir='tensorboard_logs/{:%Y-%m-%d-%H-%M}'.format(t))  # Dial

with h5py.File(DATADIR+training_file, "r") as f1:
    with h5py.File(DATADIR+label_file, "r") as f2:
        x_train = f1['training_data']  # Note that Keras is special in being able to read the HDF5 _object_
        y_train = f2['training_labels']

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle='batch',  # Required for using HDF5
                  callbacks=[tensorboard])

# serialize model to YAML
# From https://machinelearningmastery.com/save-load-keras-deep-learning-models/
model_yaml = model.to_yaml()
with open(DATADIR+trained_model, "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(DATADIR+trained_weights)
print("Saved model to disk")
