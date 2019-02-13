import h5py, json, re
from tqdm import tqdm_notebook as tqdm
import numpy as np

with open('constants.json', 'r') as f:
    constants = json.load(f)

def main(
    raw_data = 'raw_data_train.jsonl',
    status = 'training',  # testing, predicting
    filter_bool = ('type', 'comment'),
    split_regex = r' |\.',
    remove_regex = r"\'|\"|,|\.|\n|\/|&#\d\d;|\(|\)",
    tag_patterns = {'http.*\w':' <LINK> '},
    positive_labels = ('idlewords', 'apaprocki', 'pvg', 'nostrademons', 'patio11', 'carbocation', 'grellas', 'pbsd', 'dctoedt', 'tzs', 'rayiner', 'DannyBee', 'kasey_junk', 'anigbrowl', 'harryh', 'potatolicious', 'mechanical_fish'),
    sequence_length = 300,
    preprocessed_data = 'preprocessed.txt',
    label_file = 'label_and_index.h5',
):
    labels = []
    original_ids = []

    with open(constants['DATADIR']+raw_data, 'r') as infile:
        with open(constants['DATADIR']+preprocessed_data, 'w') as outfile:
            outfile.write(','.join([str(x) for x in range(sequence_length)])+'\n')
            for line in tqdm(infile):  # Dial
                if status in ('training', 'testing'):
                    line_json = json.loads(line)

                    if line_json[filter_bool[0]] != filter_bool[1]: continue
                    try:
                        temp_text = line_json['text']
                    except KeyError:
                        continue
                else:
                    temp_text = line
                temp_text = temp_text.lower()
                for key, value in tag_patterns.items():
                    temp_text = re.sub(key, value, temp_text)
                text = re.split(split_regex, re.sub(remove_regex, '', temp_text))[:sequence_length-1]
                text += ['']*(sequence_length-len(text)-1)
                text += ['\n']
                outfile.write(','.join(text))

                if status in ('training', 'testing'):
                    labels.append((1, 0) if line_json['by'] in positive_labels else (0, 1))  # Not generalizable
                    original_ids.append(line_json['id'])

    if status in ('training', 'testing'):
        with h5py.File(constants['DATADIR']+label_file, "w") as f:
            f.create_dataset('training_labels', (len(labels), 2), dtype='int', data=labels)
            f.create_dataset('ordered_keys', (len(original_ids), 1), dtype='int', data=original_ids)


        print("Positive labels: {:0.2%}".format(np.mean(np.array(labels)[:, 0])))  # Dial



if __name__ == "__main__":
    main()
