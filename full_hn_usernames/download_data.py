"""
Take manifest_filename as input, listing one data filename per line
Also take server_url, listing the base URL for those files

Download and does minimal formatting

Output raw_data_{train, test}.jsonl with the samples
"""

import bz2
from tqdm import tqdm_notebook as tqdm
import numpy as np
from requests import get
import json

with open('constants.json', 'r') as f:
    constants = json.load(f)


def main(
        manifest_filename='manifest.txt',  # Input
        server_url='https://files.pushshift.io/hackernews/',  # Input
        raw_data='raw_data'  # Output
):

    np.random.seed = 42
    stop = False
    sample_l = []
    with open(constants['DATADIR'] + manifest_filename) as infile:
        for line in tqdm(infile):  # Dial
            remote_filename = line.split()[1]
            as_bytes = get(server_url + remote_filename + '.bz2').content
            as_text = bz2.decompress(as_bytes)
            for sample in as_text.split(b'\n'):
                if not len(sample):
                    continue
                sample_l.append(sample.decode("ascii", "ignore"))
                if len(sample_l) >= (constants['SKIP_FIRST'] + constants['NUM_SAMPLES']):
                    stop = True
                if stop:
                    break
            if stop:
                break

    sample_l = sample_l[constants['SKIP_FIRST']:]
    np.random.shuffle(sample_l)

    test_ix = int(len(sample_l) * constants['test_pct'])

    with open(constants['DATADIR'] + raw_data + '_train.jsonl', 'w') as outfile:
        outfile.write('\n'.join(sample_l[test_ix:]))

    with open(constants['DATADIR'] + raw_data + '_test.jsonl', 'w') as outfile:
        outfile.write('\n'.join(sample_l[:test_ix]))


if __name__ == "__main__":
    main()