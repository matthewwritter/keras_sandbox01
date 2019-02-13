import requests, datetime

r = requests.get('https://hacker-news.firebaseio.com/v0/maxitem.json')
max_id = r.json()
print("Max ID: {}".format(max_id))

template = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
item_count = 100

dts = []

with open(DATADIR+'predict_texts.txt', 'w') as f:
    for i in tqdm(range(max_id-item_count, max_id)):
        r = requests.get(template.format(i))
        item = r.json()
        if item['type'] != 'comment': continue
        dts.append(datetime.datetime.fromtimestamp(item['time']))
        try:
            f.write('[{}] '.format(i) + item['text'].replace('\n','')+'\n')
        except KeyError:
            pass

print('{:%H:%M}'.format(dt))



import h5py, json
import preprocess_text, index_text
import pandas as pd
import keras

with open('constants.json', 'r') as f:
    constants = json.load(f)

trained_model = 'trained_model03.h5'
indexed_filename='indexed_predict.h5'
preprocessed_data = 'preprocessed_predict.txt'
preprocess_text.main(raw_data='predict_texts.txt', status='predicting', preprocessed_data = preprocessed_data)
index_text.main(preprocessed_data = preprocessed_data, indexed_filename=indexed_filename)

model = keras.models.load_model(constants['DATADIR']+trained_model)

indexed_filename='indexed_predict.h5'
with h5py.File(constants['DATADIR']+indexed_filename, "r") as index:
    preds = model.predict(index['training_data'])
with open(constants['DATADIR']+preprocessed_data, 'r') as f:
    text_lines = f.read().split('\n')

df = pd.DataFrame({'prob':preds[:, 1], 'text':text_lines[1:-1]})
high_prob = df.sort_values('prob').head(20)
for ix, row in high_prob.iterrows():
    print("[{}] {}\n".format(row.actual, row.text.replace(',', ' ').strip()))
