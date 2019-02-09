import h5py

import logging
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'
NUM_SAMPLES = 100000
SKIP_FIRST = 10000

# Input
preprocessed_data = 'preprocessed.txt'

#Knobs
pass

# Output
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'

logger= logging.getLogger()  # Dial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



with open('w2v/glove.6B.100d.txt', 'r') as f:
    weights = []
    word_to_index = {}
    i = 0
    for line in f.read().split('\n'):
        s = line.split()
        weights.append([float(x) for x in s[1:] if len(s)])
        word_to_index[s[0]] = i
        i += 1

weights = np.array(weights)

with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "w") as f:
    f.create_dataset('weights', weights.shape, dtype='f', data=weights)
    for k, v in tqdm(word_to_index.items()):
        f.create_dataset('word_to_index/{}'.format(k), (1,), dtype='int', data=v)
    f.create_dataset('metadata/max_token', (1,), dtype='int', data=weights.shape[0]+1)
    f.create_dataset('metadata/embedding_dim', (1,), dtype='int', data=weights.shape[1])
