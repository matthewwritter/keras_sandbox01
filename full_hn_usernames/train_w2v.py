import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'
NUM_SAMPLES = 100000
SKIP_FIRST = 10000

# Input
preprocessed_data = 'preprocessed.txt'

#Knobs
embedding_dim = 2

# Output
w2v_weights_and_word_index_mapping = 'w2v_weights_and_word_index_mapping.h5'

import logging
from gensim.models import Word2Vec

logger= logging.getLogger()  # Dial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class W2VIter:
    def __init__(self, texts):
        self.texts = texts
    def __iter__(self):
        for text in self.texts:
            yield [token for token in text.split(',') if token != '']

with open(DATADIR+preprocessed_data, 'r') as f:
    sequences = f.read().split('\n')
    sequence_length = len(sequences[0])
    w2viter = W2VIter(sequences)

w2v = Word2Vec(w2viter, iter=1, min_count=1, size=embedding_dim, workers=2)

weights = np.array([w2v.wv[x] for x in w2v.wv.index2word])

with h5py.File(DATADIR+w2v_weights_and_word_index_mapping, "w") as f:
    f.create_dataset('weights', weights.shape, dtype='f', data=weights)
    for word in tqdm(w2v.wv.vocab.keys()):
        f.create_dataset('word_to_index/{}'.format(word), (1,), dtype='int', data=w2v.wv.vocab[word].index)
    f.create_dataset('metadata/embedding_dim', (1,), dtype='int', data=embedding_dim)
    f.create_dataset('metadata/max_token', (1,), dtype='int', data=weights.shape[0]+1)
    f.create_dataset('metadata/sequence_length', (1,), dtype='int', data=sequence_length)
