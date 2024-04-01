# Демонстрация собственной реализации некоего алгоритма из группы линейных классификаторов
class MyLinearClassifier:
    def __init__(self):
        pass
    # Этот метод должен подобрать для обучающей выборки параметры А и В
    # прямой y=Ax+B, наилучшим образом разделяющий слонов и носорогов
    # Главный вопрос - каким алгоритмом подобрать параметры - см метод optimizer
    def fit(self, features, targets):
        self.features=features
        self.targets=targets
        self.A, self.B, self.losses = self.optimizer()

    # Наилучшим образом это так, чтобы минимизировать функцию потерь
    # В простейшем случае функция потерь может сопадать с числом ошибок классификации
    # при данных значениях А и В на обучающей выборке
    # В общем случае выбирает функции, зависящие от числа ошибок, но не линейна
    def losses(self, A,B):
        errors=0
        for i in range(0,len(self.features)):
            x=self.features[i][0]
            y=self.features[i][1]
            y_predict=A*x+B
            if y_predict>y and self.targets[i] == "слон":
                errors+=1
            if y_predict<y and self.targets[i] == "носорог":
                errors+=1
        return errors

    # В данном случае - самый тупой вариант: полный перебор значений А и В (с шагом 1)
    def optimizer(self):
        statistics = []
        for A in range(0,50):
            for B in range(-200,200):
                statistics.append([A,B,self.losses(A,B)])
        return min(statistics, key=lambda s: s[2])

    def predict(self, data):
        predictions=[]
        for d in data:
            x=d[0]
            y=d[1]
            y_predict=self.A * x + self.B
            if y>y_predict:
                predictions.append("слон")

            else:
                predictions.append("носорог")
        return predictions

if __name__ == "__main__":
    from ML01_DataSource import getData

    # Откуда-то приходят данные
    animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

    # Создание модели
    model = MyLinearClassifier()
    # Обучение модели
    model.fit(features, labels)

#Получаем тестовую выборку от доброго дяди

    # Откуда-то берут входные данные для обучения
    animals, labels_test, features_test, classes = getData("data_elephants_rhinos_1000.txt")

    predictions = model.predict(features_test)
    errors = 0
    for i in range(0, len(labels_test)):
        if predictions[i] != labels_test[i]:
            errors += 1
    print(f"точность на тестовой выборке: {1 - errors / len(predictions)}")