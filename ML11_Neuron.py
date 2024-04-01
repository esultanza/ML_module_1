import keras.layers as layers
import numpy

# Изучаем поведение нейрона
# Генерируем тестовые входные данные
data=range(-99,100)
data=[[d/10] for d in data] # unsqueeze
data=numpy.array(data) # keras ждет numpy массива
#print(data)

#layer1=layers.Dense(1, activation="linear", input_dim=1)
#layer1=layers.Dense(1, activation="sigmoid", input_dim=1)
layer1=layers.Dense(1, activation="relu", input_dim=1) #растет на положительных значениях
results=layer1(data) # при первом вызове веса случайные
# зададим предсказуемые веса
layer1.set_weights([numpy.array([[1.0]]),numpy.array([0.0])])
results=layer1(data)
#print(results)

import matplotlib.pyplot as plt
x=[d[0] for d in data] # squeeze
plt.plot(x,results)
plt.show()