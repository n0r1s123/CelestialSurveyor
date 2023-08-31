import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print(tf.config.list_physical_devices('GPU'))

import numpy
import numpy as np
import matplotlib.pyplot as pt
import tensorflow.python.framework.ops
from PIL import Image


from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255



y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(12800, activation='relu'))
model.add(Dense(12800, activation='relu'))
model.add(Dense(10, activation='softmax'))
# print(model.summary())
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# print(model.evaluate())
print(model.summary())



import time
start_time = time.time()
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
print("!!!!!!!" + str(int(time.time() - start_time)))
model.evaluate(x_test, y_test_cat)

# image = Image.open('C:\\Users\\bsolomin\\Downloads\\6.jpg')
# new_image = image.convert("1").resize((28, 28))
# new_image.show()
# img_arr = np.array(new_image)
# # img_arr = img_arr // 255
# img_arr = 1 - img_arr
# print(img_arr)
# img_arr.reshape((28, 28))
# img_arr = numpy.expand_dims(img_arr, axis=0)
# print(img_arr)
# # img_arr.expand_dims(img_arr, axis=0)
# # tensor = tensorflow.convert_to_tensor(img_arr)
# # tensor = tensorflow.reshape(tensor, shape=(28, 28))
# # print(tensor)
# prediction = model.predict(img_arr)
# print(np.argmax(prediction))
#


