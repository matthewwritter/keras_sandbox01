"""
Convert from tokenized to indexed data for use with the model training
This is the longest running process, taking about 10m for 100k samples
"""

import h5py
import numpy as np
from tqdm import tqdm_notebook as tqdm
import dask.dataframe as dd


# Input
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'
preprocessed_data = 'preprocessed.txt'
DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'

# Knobs
sequence_length = 300  # This has to be synchronized with model.py

# Output
indexed_filename = 'indexed.h5'

with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "r") as index:
    with open(DATADIR+preprocessed_data, "r") as textfile:
        text_lines = textfile.read().split('\n')
        # indexed = np.zeros(shape=(len(text_lines)-1, sequence_length))

        tokens = []
        for x in index['token_list'][()]:
            try:
                tokens.append(x.decode('ascii'))
            except:
                tokens.append('TOKENNOTFOUND')
        ix_lookup = dict([(v, i) for i, v in enumerate(tokens)])

        df = dd.read_csv(DATADIR + preprocessed_data, dtype='S')
        df2 = df.applymap(lambda x: ix_lookup.get(x, 0))
        indexed = df2.values.compute()

        # for line_ix in tqdm(range(len(text_lines))):  # tqdm doesn't like iterators
        #     wordlist = text_lines[line_ix]
        #     for word_ix, word in enumerate(wordlist.split(',')[:sequence_length]):
        #         if word in tokens:
        #             indexed[line_ix, word_ix] = ix_lookup[word]  # .decode('ascii', 'ignore')

    print(indexed[0, :5])  # Dial
    print(indexed[0, -5:])  # Dial

    with h5py.File(DATADIR+indexed_filename, "w") as f:
        f.create_dataset('training_data',
                         indexed.shape,
                         dtype='int', data=indexed)
