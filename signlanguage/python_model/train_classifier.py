import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

clean_data = []
for i, sample in enumerate(data_dict['data']):
    if len(sample) == 84:
        if i + 1 < len(data_dict['data']):  # Check if next sample exists
            clean_data.append(data_dict['data'][i + 1])  # Take next sample
  
    else:
        clean_data.append(sample)

data = np.array(clean_data, dtype=np.float32)


labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()