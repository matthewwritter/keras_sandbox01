"""
Convert from tokenized to indexed data for use with the model training
This is the longest running process, taking about 10m for 100k samples
"""

import h5py, json
import dask.dataframe as dd

with open('constants.json', 'r') as f:
    constants = json.load(f)


def main(
        w2v_weights_and_word_index_mapping='w2v_weights_and_word_index_mapping.h5',
        preprocessed_data='preprocessed.txt',
        indexed_filename='indexed.h5'
):
    with h5py.File(constants['DATADIR'] + w2v_weights_and_word_index_mapping, "r") as index:
        tokens = []
        for x in index['token_list'][()]:
            try:
                tokens.append(x.decode('ascii'))
            except:
                tokens.append('TOKENNOTFOUND')
        ix_lookup = dict([(v, i) for i, v in enumerate(tokens)])

        df = dd.read_csv(constants['DATADIR'] + preprocessed_data, dtype='S')
        df2 = df.applymap(lambda x: ix_lookup.get(x, 0))
        indexed = df2.values.compute()

    print(indexed[0, :5])  # Dial
    print(indexed[0, -5:])  # Dial

    with h5py.File(constants['DATADIR'] + indexed_filename, "w") as f:
        f.create_dataset('training_data',
                         indexed.shape,
                         dtype='int', data=indexed)


if __name__ == "__main__":
    main()
