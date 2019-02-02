# Call all the same


# Input
raw_data = 'raw_texts.txt'
trained_model = 'trained_model.yaml'
trained_weights = 'trained_weights.h5'
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'


# Knobs
status = 'training'
filter_bool = ('type', 'comment')
split_regex = r' |\.'
remove_regex = r"\'|\"|,|\.|\n|\/|&#\d\d;|\(|\)"
tag_patterns = {'http.*\w':' <LINK> '}
sequence_length = 300

# Output
pass  # Print

import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'

from keras.models import model_from_yaml, load_model


actual_labels = []
original_ids = []
outfile_text = ''

with open(DATADIR+raw_data, 'r') as infile:
    for line in tqdm(infile):  # Dial
        temp_text = line.lower()
        for key, value in tag_patterns.items():
            temp_text = re.sub(key, value, temp_text)
        text = re.split(split_regex, re.sub(remove_regex, '', temp_text))
        text += ['']*(sequence_length-len(text))
        text += ['\n']
        outfile_text += ','.join(text)

with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "r") as index:
    text_lines = outfile_text.split('\n')
    sequence_length = index['metadata']['sequence_length'][0]
    indexed = np.zeros(shape=(len(text_lines)-1, sequence_length))
    keys = index['word_to_index'].keys()
    for line_ix, wordlist in tqdm(enumerate(text_lines), total=len(text_lines)):
        for word_ix, word in enumerate(wordlist.split(',')[:sequence_length]):
            if word in keys:
                indexed[line_ix, word_ix] = index['word_to_index'][word][()]
        break

from keras.models import model_from_yaml

# load YAML and create model
with open(DATADIR+trained_model, 'r') as yaml_file:
    loaded_model_yaml = yaml_file.read()
    loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights(DATADIR+trained_weights)
print("Loaded model from disk")

preds = loaded_model.predict(indexed)
preds.max(axis=0)

import pandas as pd
df = pd.DataFrame({'prob':preds[:, 1], 'text':text_lines[:-1]})
high_prob = df.sort_values('prob', ascending=False).head(20)
for ix, row in high_prob.iterrows():
    print("{}\n".format(row.text.replace(',', ' ').strip()))
