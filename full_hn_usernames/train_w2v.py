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
pass  # Now that we're downloading, there are no knobs

# Output
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'

logger= logging.getLogger()  # Dial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



with open('/home/mritter/code/twitter_nlp/w2v/glove.6B.100d.txt', 'r') as f:
    weights = []
    token_list = []
    for line in tqdm(f.read().split('\n')):
        s = line.split()
        if len(s):
            weights.append([float(x) for x in s[1:]])
            token_list.append(s[0].encode('utf8'))

weights = np.array(weights)
token_list = np.array(token_list, dtype='S')

with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "w") as f:
    f.create_dataset('weights', weights.shape, dtype='f', data=weights)
    f.create_dataset('token_list', (len(token_list),), data=token_list)  # Numpy alraedy provided a dtype
    f.create_dataset('metadata/max_token', (1,), dtype='int', data=weights.shape[0]+1)
    f.create_dataset('metadata/embedding_dim', (1,), dtype='int', data=weights.shape[1])
