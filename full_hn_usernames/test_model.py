import h5py, json
import preprocess_text, index_text
import pandas as pd
import keras

with open('constants.json', 'r') as f:
    constants = json.load(f)

trained_model = 'trained_model03.h5'
label_file = 'label_and_index_test.h5'
indexed_filename='indexed_test.h5'
preprocessed_data = 'preprocessed_test.txt'
preprocess_text.main(raw_data='raw_data_test.jsonl', status='testing', preprocessed_data = preprocessed_data, label_file = label_file)
index_text.main(preprocessed_data = 'preprocessed_test.txt', indexed_filename=indexed_filename)

model = keras.models.load_model(constants['DATADIR']+trained_model)

indexed_filename='indexed_test.h5'
with h5py.File(constants['DATADIR']+indexed_filename, "r") as index:
    preds = model.predict(index['training_data'])
with h5py.File(constants['DATADIR']+label_file, "r") as index:
    labels = index['training_labels'][()]
with open(constants['DATADIR']+preprocessed_data, 'r') as f:
    text_lines = f.read().split('\n')

df = pd.DataFrame({'prob':preds[:, 1], 'actual': labels[:, 0], 'text':text_lines[1:-1]})
high_prob = df.sort_values('prob').head(20)
for ix, row in high_prob.iterrows():
    print("[{}] {}\n".format(row.actual, row.text.replace(',', ' ').strip()))
