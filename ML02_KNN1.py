from ML01_DataSource import getData

# Демонстрация алгоритма KNN
animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

# Гиперпараметр модели (=число соседей для определения)
k = 5

# Большинство алгоритмов МО требуют масштабирования (scale) входных признаков
scale_age = 100
scale_weight = 2700

features = [[l[0]/scale_age,l[1]/scale_weight] for l in features]
#print(features)

# Реализуем KNN
def predict(age, weight):
    distance = []
    # Определяем расстояние до каждой из точек обучающей выборки
    for i in range(0, len(features)):
        x = features[i][0]
        y = features[i][1]
        # Манхэттенская метрика
        dist = abs(age-x) + abs(weight-y)
        distance.append([dist, labels[i]])
    # Пересчитаем, сколько носорогов в ближайших пяти
    distance.sort(key=lambda d: d[0])
    distance = distance[0:k]
    #print(distance)

    # Посчитаем число элементов класса 1 (носорогов) среди этих 5
    n0 = len(list(filter(lambda d: d[1] == classes[0], distance)))
    #print(n1)
    if n0>k/2:
        return classes[0]
    else:
        return classes[1]

if __name__ == "__main__":
    print(predict(50/scale_age, 1200/scale_weight))
    print(predict(90/scale_age, 2500/scale_weight))
    print(predict(80/scale_age, 1000/scale_weight))