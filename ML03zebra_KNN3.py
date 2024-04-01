from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from ML01_DataSource import getData

#files=["data_elephants_zebras_90.txt", "data_elephants_zebras_10.txt"]
files=["data_animals_150.txt", "data_animals_150.txt"]
# Использование стандартной библиотеки sklearn (scilearn) для применения алгоритма KNN

animals, labels, features, classes = getData(files[0])

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


animals, labels_test, features_test, classes = getData(files[1])
#Масштабируем так же, как обучающую
features_test=scaler.transform(features_test)
predictions=model.predict(features_test)
errors=0

for i in range (0, len(labels_test)):
    if predictions[i] !=labels_test[i]:
        errors +=1
print(f"точность на тестовой выборке: {1-errors/len(predictions)}")

errors=[]
for c in classes:
    errors.append({"name": c, "error": 0})

for i in range (0, len(labels_test)):
    if predictions[i] !=labels_test[i]:
        for e in errors:
            if labels_test[i]==e["name"]:
                e["error"] +=1

for e in errors:
    print(f"Точность по классу {e['name']}:{1-e['error']/len(list(filter(lambda l: l==e['name'], labels_test)))}")
    #или /labels.count(e['name'])}")

#print(f"Ошибки по классам на тестовой выборке: {1-errors/len(predictions)}")

#проверка обученной модели
#print(model.predict([[50, 490]]))