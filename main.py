from tensorflow import keras
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.regularizers import l2

# Load datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bring images to the desired format
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /=  255

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

validation_x = x_train[50000:]
validation_y = y_train[50000:]

x_train = x_train[:50000]
y_train = y_train[:50000]

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
validation_y = keras.utils.to_categorical(validation_y, 10)

# Create model
model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.0001)),
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train and save model
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_data=(validation_x, validation_y))
model.save('../saved_models/model')

# Load model and test it on the test dataset
# model_new = keras.models.load_model('../saved_models/model')
# score_best = model_new.evaluate(x_test, y_test_cat)
# print(f'Loss: {score_best[0]}')
# print(f'Accuracy: {score_best[1]}')
