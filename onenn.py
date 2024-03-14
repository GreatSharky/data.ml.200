from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time

start = time.time()
data_train = fetch_20newsgroups(subset='test')
data_test = fetch_20newsgroups(subset='train')


vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target
X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print("Simple baseline:", dummy_clf.score(X_test,y_test), "\nTime (seconds):", time.time()-start)

#def norms(X):
 #   X_test_norms = []
  #  for Xi in X:
   #     Xi = Xi.toarray()
    #    norm = np.linalg.norm(Xi)
    #    X_test_norms.append(norm)
    #return np.array(X_test_norms)

#X_test_norms = norms(X_test)
#X_train_norms = norms(X_train)

output = []
try:
    t = time.time()
    for i, Xi in enumerate(X_test):
        Xi = Xi.toarray()
        min_dis = 1E20
        classification = None
        
        for j, Xj in enumerate(X_train):
            Xj.toarray()
            norm = np.linalg.norm(Xi-Xj)
            if norm < min_dis:
                classification = y_train[j]
                min_dis = norm
        output.append(classification)
except KeyboardInterrupt:
    t1 = time.time()
    print(f"{len(output)} points evaluated. It took {t1-t} seconds.")
    print(f"{len(output)/X_test.shape[0]} percent evaluated. To evaluate all points would take approximetly {(t1-t)/len(output)*X_test.shape[0]/60} minutes.")

y_pred = np.array(output)
print("My NN1 prediction score:",accuracy_score(y_test[:len(output)], y_pred))
print("Time from start:", time.time()-start)
print("Number of samples:", len(y_pred))

sk_NN1 = NearestNeighbors(n_neighbors=1)

sk_NN1.fit(X_train)
y_indexes = sk_NN1.kneighbors(X=X_test, return_distance=False)

y_pred = []
for i in y_indexes:
    y_pred.append(y_train[i])

print("SK NN1 prediction socre:", accuracy_score(y_pred,y_test))
print("Time from start:", time.time() - start)