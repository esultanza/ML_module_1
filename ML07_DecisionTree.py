# Использование алгоритма семейства DecisionTree

from sklearn.tree import DecisionTreeClassifier
from ML01_DataSource import getData

#files=["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]
#files=["data_animals_150.txt", "data_animals_150.txt"]
files=["data_elephants_rhinos_1000.txt","data_elephants_rhinos_100.txt"]

# Использование стандартной библиотеки sklearn (scikit-learn) для применения алгоритма KNN

animals, labels, features, classes = getData(files[0])

# Создаем и обучаем модель
model=DecisionTreeClassifier(max_depth=50) #только эту строку изменить по сравнению с методом из ML06
model.fit(features, labels)

#Проверяем точность на тестовой выборке
predictions=model.predict(features)

errors=0
for i in range (0, len(labels)):
    if predictions[i] !=labels[i]:
        errors +=1
print(f"точность на обучающей выборке: {1-errors/len(predictions)}")


animals, labels_test, features_test, classes = getData(files[1])
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
    print(f"Точность по классу {e['name']}:{1-e['error']/labels_test.count(e['name'])}")

# Дерево решений интересно тем, что обученную модель можно визуализировать
from sklearn import tree
text_representation = tree.export_text(model)
#print(text_representation)