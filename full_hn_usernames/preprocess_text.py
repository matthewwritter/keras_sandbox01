import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'

# Input
raw_data = 'raw_data_train.jsonl'

# Knobs
status = 'training'
filter_bool = ('type', 'comment')
split_regex = r' |\.'
remove_regex = r"\'|\"|,|\.|\n|\/|&#\d\d;|\(|\)"
tag_patterns = {'http.*\w':' <LINK> '}
positive_labels = ('idlewords', 'apaprocki', 'pvg', 'nostrademons', 'patio11', 'carbocation', 'grellas', 'pbsd', 'dctoedt', 'tzs', 'rayiner', 'DannyBee', 'kasey_junk', 'anigbrowl', 'harryh', 'potatolicious', 'mechanical_fish')
sequence_length = 300

# Output
preprocessed_data = 'preprocessed.txt'
label_file = 'label_and_index.h5'

import re

labels = []
original_ids = []

with open(DATADIR+raw_data, 'r') as infile:
    with open(DATADIR+preprocessed_data, 'w') as outfile:
        outfile.write(','.join([str(x) for x in range(sequence_length)])+'\n')
        for line in tqdm(infile):  # Dial
            line_json = json.loads(line)

            if line_json[filter_bool[0]] != filter_bool[1]: continue
            try:
                temp_text = line_json['text']
            except KeyError:
                continue
            temp_text = temp_text.lower()
            for key, value in tag_patterns.items():
                temp_text = re.sub(key, value, temp_text)
            text = re.split(split_regex, re.sub(remove_regex, '', temp_text))[:sequence_length-1]
            text += ['']*(sequence_length-len(text)-1)
            text += ['\n']
            outfile.write(','.join(text))

            if status == 'training':
                labels.append((1, 0) if line_json['by'] in positive_labels else (0, 1))  # Not generalizable
                original_ids.append(line_json['id'])

if status == 'training':
    with h5py.File(DATADIR+label_file, "w") as f:
        f.create_dataset('training_labels', (len(labels), 2), dtype='int', data=labels)
        f.create_dataset('ordered_keys', (len(original_ids), 1), dtype='int', data=original_ids)


print("Positive labels: {:0.2%}".format(np.mean(np.array(labels)[:, 0])))  # Dial