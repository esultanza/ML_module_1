# Задача кластеризации разделить на группы входные данные.
# Иногда это называют обучение без учителя
from sklearn.cluster import KMeans
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

import sklearn.datasets as ds
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy
from numpy import ndarray
from sklearn.preprocessing import StandardScaler

irises = ds.load_iris()
# print(irises)
features: ndarray = irises.data
# print(features)
targets: ndarray = irises.target
# print(targets)

# Шкалирование
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

# Классифицируем
# model=KNeighborsClassifier()
# model.fit(features, targets)

model=KMeans(n_clusters=2)
model.fit(features)
predictions = model.predict(features)

# Визуализация результатов (выдает маленьких и больших вместо слонов и зебр)
import matplotlib.pyplot as pplt

for i in range(0,len(features)):
    x = features[i][0]
    y=features[i][1]
    if predictions[i]==0:
        pplt.plot(x,y,".r")
    else:
        pplt.plot(x,y,"ob")
pplt.show()