# Input
compiled_model = 'trained_model03.h5'
training_file = 'indexed.h5'
label_file = 'label_and_index.h5'

# Knobs
epochs = 50
batch_size = 128

# Output
trained_model = 'trained_model04.h5'


import h5py
from keras.models import load_model
from keras.callbacks import TensorBoard as tb
from datetime import datetime

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'


model = load_model(DATADIR+compiled_model)
print("Loaded model from disk")


t = datetime.now()
tensorboard = tb(log_dir='/home/mritter/code/twitter_nlp/tensorboard_logs/{:%Y-%m-%d-%H-%M}'.format(t))  # Dial

with h5py.File(DATADIR+training_file, "r") as f1:
    with h5py.File(DATADIR+label_file, "r") as f2:
        x_train = f1['training_data']  # Note that Keras is special in being able to read the HDF5 _object_
        y_train = f2['training_labels']

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle='batch',  # Required for using HDF5
                  callbacks=[tensorboard])


print("training complete")

# Shot in the dark to reduce potential memory issues
# del x_train
# del y_train


model.save(DATADIR+trained_model)
print("Saved model to disk")
