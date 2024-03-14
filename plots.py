import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups

test_data = fetch_20newsgroups(subset="test")
y_test = test_data.target


data = []
with open("data.txt", "r") as file:
    for row in file:
        data.append([int(x) for x in row.split(";")])
print(len(data))

stops_data = data[:25]
no_stops_data = data[25:]

stops_acc = []
#for point in stops_data:
#    stops_acc.append(accuracy_score(point, y_test))
#for point in no_stops_data:
#    stops_acc.append(accuracy_score(point, y_test))

fig, ax = plt.subplots()
ax.plot(len(stops_acc), stops_acc)

plt.show()