import h5py, json, re
import preprocess_text, index_text
import pandas as pd
import keras

import requests, datetime

with open('constants.json', 'r') as f:
    constants = json.load(f)

trained_model = 'trained_model03.h5'
indexed_filename='indexed_predict.h5'
preprocessed_data = 'preprocessed_predict.txt'
predict_texts = 'predict_texts.txt'
live_predictions = 'live_predictions.csv'

r = requests.get('https://hacker-news.firebaseio.com/v0/maxitem.json')
max_id = r.json()
print("Max ID: {}".format(max_id))

template = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
item_count = 100

dts = []
ids = []
with open(constants['DATADIR']+predict_texts, 'w') as f:
    for i in range(max_id-item_count, max_id):
        r = requests.get(template.format(i))
        item = r.json()
        if item['type'] != 'comment': continue
        dts.append(datetime.datetime.fromtimestamp(item['time']))
        try:
            # f.write('[{}] '.format(i) + item['text'].replace('\n',' ')+'\n')
            f.write(item['text'].replace('\n',' ')+'\n')
            ids.append(i)
        except KeyError:
            pass

preprocess_text.main(raw_data=predict_texts, status='predicting', preprocessed_data = preprocessed_data)
index_text.main(preprocessed_data = preprocessed_data, indexed_filename=indexed_filename)

model = keras.models.load_model(constants['DATADIR']+trained_model)

indexed_filename='indexed_predict.h5'
with h5py.File(constants['DATADIR']+indexed_filename, "r") as index:
    preds = model.predict(index['training_data'])
with open(constants['DATADIR']+preprocessed_data, 'r') as f:
    text_lines = f.read().split('\n')
# with open(constants['DATADIR']+predict_texts, 'r') as f:
#     extracted = [re.findall(r'\[(\d+)\]', x) for x in f.read().split('\n')]
#     ids = [x[0] if len(x) else 'NA' for x in extracted]

df = pd.DataFrame({'prob':preds[:, 1], 'text':text_lines[1:-1], 'id':ids})
high_prob = df.sort_values('prob', ascending=False).head(20)
df.to_csv(constants['DATADIR']+live_predictions)
for ix, row in high_prob.iterrows():
    print("{} [https://news.ycombinator.com/item?id={}]\n".format(row.text.replace(',', ' ').strip(), row.id))

