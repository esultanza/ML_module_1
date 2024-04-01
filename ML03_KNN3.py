from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from ML01_DataSource import getData

# Использование стандартной библиотеки sklearn (scilearn) для применения алгоритма KNN

animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

# Масштабируем стандартным скейлером
scaler=MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
#print(features)

# Создаем и обучаем модель
model=KNeighborsClassifier(5)
model.fit(features, labels)

#Проверяем точность на тестовой выборке
predictions=model.predict(features)

errors=0
for i in range (0, len(labels)):
    if predictions[i] !=labels[i]:
        errors +=1
print(f"точность на обучающей выборке: {1-errors/len(predictions)}")

animals, labels_test, features_test, classes = getData("data_elephants_rhinos_1000.txt")
#Масштабируем так же, как обучающую
features_test=scaler.transform(features_test)
predictions=model.predict(features_test)
errors=0

for i in range (0, len(labels_test)):
    if predictions[i] !=labels_test[i]:
        errors +=1
print(f"точность на тестовой выборке: {1-errors/len(predictions)}")

# 1.слоны и зебры
# 2. переписать для n>2 классификаторов в файлы KNN2