import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D , Flatten, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from sklearn.metrics import accuracy_score,confusion_matrix

class CNN:
    def cnn_model1(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model2(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model3(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model4(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model5(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model6(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model7(self):
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model8(self):
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        return model

    def cnn_model9(self):
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='softmax'))

        return model





