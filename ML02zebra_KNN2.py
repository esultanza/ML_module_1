# Реализуем алгоритм KNN в стиле библиотек МО
# Делаем класс модели
class KNN:
    # Гиперпараметры модели передаются через конструктор модели
    def __init__(self, k, metrics = "manhattan"):
        self.k = k
        self.metrics = metrics

    # Метод, реализующй обучение модели (fit - подгонка)
    # В KNN очень просто: надо запомнить обучающую выборку
    # В общем случае здесь может быть очень сложный вычисляющий параметры модели алгоритм
    def fit(self, features, labels):
        self.features=features
        self.labels=labels
        self.classes=list(set(labels)).sort()

    # Реализуем KNN
    def predictOne(self, age, weight):
        distance = []
        # Определяем расстояние до каждой из точек обучающей выборки
        for i in range(0, len(self.features)):
            x = self.features[i][0]
            y = self.features[i][1]
            if self.metrics == "manhattan":
                # Манхэттенская метрика
                dist = abs(age-x) + abs(weight-y)
            else:
                # Эвклидова метрика
                dist = ((age - x)**2 + (weight - y)**2)**0.5
            distance.append([dist, labels[i]])
        # Пересчитаем, сколько носорогов в ближайших пяти
        distance.sort(key=lambda d: d[0])
        distance = distance[0:self.k]
        #print(distance)

        # Посчитаем число элементов класса 1 (носорогов) среди этих 5
        n0 = len(list(filter(lambda d: d[1] == classes[0], distance)))
        # Надо бы переписать для n>2 классификаторов
        if n0>self.k/2:
            return classes[0]
        else:
            return classes[1]

    def predict(self, data):
        results=[]
        for item in data:
            results.append(self.predictOne(item[0], item[1]))
        return results

if __name__ == "__main__":
    from ML01_DataSource import getData

    # Откуда-то приходят данные
    animals, labels, features, classes = getData("data_elephants_zebras_90.txt")
    # Большинство алгоритмов МО требуют масштабирования (scale) входных значений
    # В общем случае здесь может быть сложный препроцессинг
    scale_age = 100
    scale_weight = 2400

    features = [[l[0] / scale_age, l[1] / scale_weight] for l in features]

    # Создание модели
    model = KNN(3)
    # Обучение модели
    model.fit(features, labels)

    # Предварительная проверка обученной модели
    print(model.predictOne(65 / scale_age, 273 / scale_weight))
    print(model.predictOne(90 / scale_age, 470 / scale_weight))
    print(model.predictOne(5 / scale_age, 200 / scale_weight))
    # print(model.predict([
    #     [50 / scale_age, 1200 / scale_weight],
    #     [90 / scale_age, 2500 / scale_weight],
    #     [80 / scale_age, 1000 / scale_weight]]))

    # Проверить точность предсказаний модели на тестовой выборке (testing set)
    # 1. Не вполне корректный вариант с использованием обучающей выборки в качестве тестовой

    predictions=model.predict(features)
    errors=0
    for i in range (0, len(labels)):
        if predictions[i] !=labels[i]:
            errors +=1
    print(f"точность на обучающей выборке: {1-errors/len(predictions)}")


    # 2. Использовать для обучения входную выборку за вычетом некоей случайной ее подвыборки.
    # Случайную подвыборку использовать в качестве тестовой - не делаем

    # 3. Получаем тестовую выборку от доброго дяди

    # Откуда-то берут входные данные для обучения
    animals, labels_test, features_test, classes = getData("data_elephants_zebras_10.txt")

    # Масштабируем так же, как и обучающую
    scale_age = 100
    scale_weight = 2400

    features_test = [[l[0] / scale_age, l[1] / scale_weight] for l in features_test]

    predictions = model.predict(features_test)
    errors = 0
    for i in range(0, len(labels_test)):
        if predictions[i] != labels_test[i]:
            errors += 1
    print(f"точность на тестовой выборке: {1 - errors / len(predictions)}")

