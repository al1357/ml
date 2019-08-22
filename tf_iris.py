import data_loader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

dl = data_loader.data_loader([0.3, 0])
dl.load_iris()

x_train = dl.get_x(mod="scaled").T
#x_train = dl.get_x().T
y_train = dl.get_y().T

x_cv = dl.get_x("cv", mod="scaled").T
#x_cv = dl.get_x("cv").T
y_cv = dl.get_y("cv").T

model = keras.Sequential([
            keras.layers.Dense(4),
            keras.layers.Dense(24, activation=tf.nn.tanh),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=10)

ev_res = model.evaluate(x_cv, y_cv, verbose=1)
print(model.metrics_names)
print(ev_res)