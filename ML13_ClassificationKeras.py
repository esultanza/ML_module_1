import keras.layers as layers
import keras.models as models
import numpy
from keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import StandardScaler
from ML01_DataSource import getData

# files=["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]
# files=["data_animals_150.txt", "data_animals_150.txt"]
files = ["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]

# Использование стандартной библиотеки sklearn (scikit-learn) для применения алгоритма KNN

animals, y_train, x_train, classes = getData(files[0])
# Для этого алгоритма шкалирование обязательно
scaler = StandardScaler().fit(x_train)
x_train = numpy.array(scaler.transform(x_train))
# Для нейронных сетей у (целевой вектор) должны быть числами
# Это называет проблемой кодировки категориальных признаков
y_train = numpy.array([0 if y == "носорог" else 1 for y in y_train])
# print(y_train)

# Простая однонейронная сеть (есть мнение, что это эквивалент логистической регрессии:
# по равенству точности на обучающей и тестовой выборках)
# model = models.Sequential()
# model.add(layers.Dense(1, activation="sigmoid", input_dim=2))

# Выстроим более сложные сети
model = models.Sequential()
model.add(layers.Dense(8, activation="relu", input_dim=2))
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))  # 1 - потому что он последний, а классификация бинарная

model.compile(
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=0)
# accuracy может быть каждый раз разной из-за отсутствия определенных весов

# Проверим точность на тестовой выборке
animals, y_test, x_test, classes = getData(files[1])
# Для этого алгоритма шкалирование обязательно
x_test = numpy.array(scaler.transform(x_test))
# Для нейронных сетей у (целевой вектор) должны быть числами
# Это называет проблемой кодировки категориальных признаков
y_test = numpy.array([0 if y == "носорог" else 1 for y in y_test])

# К этому моменту модель уже обучена
model.evaluate(x_test, y_test)

