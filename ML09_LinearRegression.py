# Задача регрессии - экстраполировать или интерполировать (аппроксимировать) числовые данные.
# Например, сделать модель, которая предсказывает вес слона по его возрасту

from sklearn.linear_model import LinearRegression
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

#files=["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]
#files=["data_animals_150.txt", "data_animals_150.txt"]
files=["data_elephants_rhinos_1000.txt"]

# Использование стандартной библиотеки sklearn (scikit-learn) для применения алгоритма KNN

animals, labels, features, classes = getData(files[0])
# оставляем только данные о слонах
features=features[0:500]

x=[[d[0]] for d in features]
y=[[d[1]] for d in features]
# print(x)
# Создаем и обучаем модель
model=LinearRegression()
model.fit(x,y)

#Определяем вес слона по его возрасту
predictions=model.predict([[50]])
print(predictions)