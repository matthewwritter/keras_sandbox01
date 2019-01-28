import h5py, bz2, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

DATADIR = '/home/mritter/code/twitter_nlp/full_hn_usernames/data/'
NUM_SAMPLES = 100000
SKIP_FIRST = 10000

# Input
manifest_filename = 'manifest.txt'
server_url = 'https://files.pushshift.io/hackernews/'

# Knobs
test_pct = .5

# Output
raw_data = 'raw_data'


from requests import get

np.random.seed = 42
stop = False
sample_l = []
with open(DATADIR+manifest_filename) as infile:
    for line in tqdm(infile):  # Dial
        remote_filename = line.split()[1]
        as_bytes = get(server_url+remote_filename+'.bz2').content
        as_text = bz2.decompress(as_bytes)
        for sample in as_text.split(b'\n'):
            if not len(sample): continue
            sample_l.append(sample.decode("ascii", "ignore"))
            if len(sample_l) >= (SKIP_FIRST + NUM_SAMPLES):
                stop = True
            if stop: break
        if stop: break

sample_l = sample_l[SKIP_FIRST:]
np.random.shuffle(sample_l)

test_ix = int(len(sample_l)*test_pct)

with open(DATADIR+raw_data+'_train.jsonl', 'w') as outfile:
    outfile.write('\n'.join(sample_l[test_ix:]))

with open(DATADIR+raw_data+'_test.jsonl', 'w') as outfile:
    outfile.write('\n'.join(sample_l[:test_ix]))
