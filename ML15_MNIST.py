import numpy
from keras.datasets import mnist
import keras.models as models
import keras.layers as layers
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
# Получаем и Исследуем данные из MNIST
(train_x, train_y), (test_x,test_y) = mnist.load_data()
# print(len(train_x))
train_x=train_x/255
test_x=test_x/255
picture=train_x[123] #выводит попиксельно с яркостью, 28 пикселей в длину
#print(picture)
# print(train_y[123])
# plt.imshow(picture, cmap="Greys", interpolation="nearest") #Отрисовка картинки
# plt.show()
# exit()
# Получаем данные из файла
picture=cv2.imread("data/T1210.jpg")
picture=numpy.array(picture)/255
#print(picture)

model = models.Sequential()
# сверточный слой
#model.add(layers.Conv2D(input_shape=(12,10,3), kernel_size=(3,3), filters=3))
model.add(layers.MaxPool2D(strides=2))

result=model.layers[0](numpy.array([picture]))[0] # unsqueeze > squeeze
plt.imshow(result) #Отрисовка картинки
plt.show()
exit()

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28))) # распрямляем матрицу в один ряд
# layer=model.layers[0]
# result=layer(numpy.array([picture]))
# print(result[0])

model.add(layers.Dense(100,activation="relu")) # 100 - это здравый смысл, показывающий, что число признаков, по которым можно различать цифры
model.add(layers.Dense(10, activation="sigmoid"))# или softmax
model.compile(
    loss="sparse_categorical_crossentropy", # categorical_crossentropy годится только для one-hot
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=5)
model.evaluate(test_x,test_y)