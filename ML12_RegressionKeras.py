import keras.layers as layers
import keras.models as models
from keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import MinMaxScaler
import math

# Предсказываем значение в числовом ряду
# Изучаем поведение нейрона
# Генерируем тестовые входные данные
data = range(0, 100)
x = [ [d / 10] for d in data]
# y = [ [d ** 2] for d in data ] # x квадрат
y = [ [math.exp(d)] for d in data ] # "резкая" экспонента

# Шкалируем входные данные
scalerX=MinMaxScaler().fit(x)
scalerY=MinMaxScaler().fit(y)
x=scalerX.transform(x)
y=scalerY.transform(y)
# print(x)
# print(y)

# Однослойная модель однонейронная модель не может выдать что-то иного кроме линейной функции
# model=models.Sequential()
# model.add(layers.Dense(1,activation="linear", input_dim=1))

# Многослойная сеть (Для аппроксимации квадратов нормально)
model=models.Sequential()
model.add(layers.Dense(100,activation="relu", input_dim=1))
model.add(layers.Dense(1,activation="linear"))

# Многослойная сеть (Для аппроксимации резкой экспоненты)
# Хороший результат: 100 нейронов, learning_rate = 0.01, 1000 эпох
# Решающий параметр при этом: batch_size = 100
# Отличный результат: 100 нейронов, learning_rate = 0.01, 5000 эпох
# Экспериментально: кастомная функция потерь,


model=models.Sequential()
model.add(layers.Dense(100,activation="relu", input_dim=1))
model.add(layers.Dense(1,activation="linear")) # Второй слой на основе нейронов предыдущего

import keras.backend as K
def max_abs_error(y_true, y_pred):
    print(y_true)
    return K.max(K.abs(y_true - y_pred))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error" #среднее квадратичное отклонение
    # loss=max_abs_error
)
model.fit(x,y, epochs=10000, verbose=0, batch_size=100)
predictions = model.predict(x)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.plot(x,predictions)
plt.show()