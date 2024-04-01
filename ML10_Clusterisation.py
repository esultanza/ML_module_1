# Задача кластеризации разделить на группы входные данные.
# Иногда это называют обучение без учителя
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, Birch
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

files=["data_elephants_zebras_90.txt"]

animals, labels, features, classes = getData(files[0])
# Для этого алгоритма шкалирование обязательно
scaler=StandardScaler().fit(features)
features=scaler.transform(features)
# Создаем и обучаем модель
model=KMeans(n_clusters=2)
model.fit(features)

predictions=model.predict(features)
#print(predictions)

# Визуализация результатов (выдает маленьких и больших вместо слонов и зебр)
import matplotlib.pyplot as pplt

for i in range(0,len(features)):
    x = features[i][0]
    y=features[i][1]
    if predictions[i]==0:
        pplt.plot(x,y,"or")
    else:
        pplt.plot(x,y,"ob")
pplt.show()