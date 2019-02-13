import h5py, json
import preprocess_text, index_text
import pandas as pd
import keras

with open('constants.json', 'r') as f:
    constants = json.load(f)

# File inputs and outputs
trained_model = 'trained_model03.h5'
preprocessed_data = 'preprocessed_test.txt'

label_file = 'label_and_index_test.h5'
indexed_filename='indexed_test.h5'
with h5py.File(constants['DATADIR'] + label_file, "r") as index:
    labels = index['training_labels'][()]
print("labels loaded")

# Run preprocessing and indexing
preprocess_text.main(raw_data='raw_data_test.jsonl', status='testing', preprocessed_data = preprocessed_data, label_file = label_file)
print("preprocessed")
index_text.main(preprocessed_data = 'preprocessed_test.txt', indexed_filename=indexed_filename)
print("indexed")

# Make predictions
with h5py.File(constants['DATADIR']+indexed_filename, "r") as index:
    model = keras.models.load_model(constants['DATADIR'] + trained_model)
    print("model loaded")
    preds = model.predict(index['training_data'])
    print("predictions made")

# Display results
with open(constants['DATADIR']+preprocessed_data, 'r') as f:
    text_lines = f.read().split('\n')
print("text loaded")
df = pd.DataFrame({'prob':preds[:, 1], 'actual': labels[:, 0], 'text':text_lines[1:-1]})
print("df made")
high_prob = df.sort_values('prob').head(20)

for ix, row in high_prob.iterrows():
    print("[{}] {}\n".format(row.actual, row.text.replace(',', ' ').strip()))
