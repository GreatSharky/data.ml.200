from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

y_train = train_data.target
y_test = test_data.target


def create_data(train_data, test_data, N, use_stop_words=True):
    data = []
    if use_stop_words:
        word = "english"
    else:
        word = None
    for n in N:
        print(n, word)
        vectorizer = CountVectorizer(max_features=n, stop_words=word)
        X_train = vectorizer.fit_transform(train_data)
        X_test = vectorizer.transform(test_data)
        sknn = NearestNeighbors(n_neighbors=1)
        sknn.fit(X_train)
        y_indexes = sknn.kneighbors(X=X_test, return_distance=False)
        ypred = []
        for i in y_indexes:
            ypred.append(y_train[i])
        
        data.append(ypred)
    return data

N = [i*100 for i in range(1,25)]
stops = create_data(train_data.data, test_data.data, N)
no_stops = create_data(train_data.data, test_data.data, N, False)
print(len(stops))

data_file = "data.txt"
with open(data_file, "w") as file:
    for data in stops:
        file.write(";".join(str(x).strip("[").strip("]") for x in data)+"\n")
    for data in no_stops:
        file.write(";".join(str(x).strip("[").strip("]") for x in data)+"\n")

