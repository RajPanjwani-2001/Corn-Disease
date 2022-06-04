import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D , Flatten, MaxPool2D, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16

import pickle
import numpy as np
import pandas as pd
from config import config
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


fp = open('Combined_2.pkl', "rb")
features = pickle.load(fp)
fp.close()

fp = open('ClassLabels_2.pkl', "rb")
cls_labels = pickle.load(fp)
fp.close()


blight = []
blight_cls = []
common_rust = []
common_rust_cls = []
gray_leaf = []
gray_leaf_cls = []
healthy = []
healthy_cls = []

for i in range(cls_labels.shape[0]):
    if cls_labels[i] == 0:
        blight.append(features[i,:])
        blight_cls.append(0)

    if cls_labels[i] == 1:
        common_rust.append(features[i, :])
        common_rust_cls.append(1)

    if cls_labels[i] == 2:
        gray_leaf.append(features[i, :])
        gray_leaf_cls.append(2)

    if cls_labels[i] == 3:
        healthy.append(features[i,:])
        healthy_cls.append(3)


blight = np.array(blight)
blight_cls = np.array(blight_cls)

common_rust = np.array(common_rust)
common_rust_cls = np.array(common_rust_cls)

gray_leaf = np.array(gray_leaf)
gray_leaf_cls = np.array(gray_leaf_cls)

healthy = np.array(healthy)
healthy_cls = np.array(healthy_cls)

print(blight.shape, common_rust.shape, gray_leaf.shape, healthy.shape)

Path = 'C:/Users/Raj/MyProg/Corn_Disease/codes/train_index/'
train_pickle = ['blight','common_rust','gray_leaf','healthy']

bl = Path+train_pickle[0]+'_train_index_2'+".pkl"
cr = Path+train_pickle[1]+'_train_index_2'+".pkl"
gl = Path+train_pickle[2]+'_train_index_2'+".pkl"
hl = Path+train_pickle[3]+'_train_index_2'+".pkl"

fp = open(bl, "rb")
blight_train_index = pickle.load(fp)
fp.close()

fp = open(cr, "rb")
common_rust_train_index = pickle.load(fp)
fp.close()

fp = open(gl, "rb")
gray_leaf_train_index = pickle.load(fp)
fp.close()

fp = open(hl, "rb")
healthy_train_index = pickle.load(fp)
fp.close()

fo = open('model9_results.csv','a')
for i in range(6):
    X_train, X_test = blight[blight_train_index[i][0], :], blight[blight_train_index[i][1], :]
    Y_train, Y_test = blight_cls[blight_train_index[i][0]], blight_cls[blight_train_index[i][1]]

    train, test = common_rust[common_rust_train_index[i][0], :], common_rust[common_rust_train_index[i][1], :]
    X_train = np.append(X_train, train, axis=0)
    X_test = np.append(X_test, test, axis=0)
    train, test = common_rust_cls[common_rust_train_index[i][0]], common_rust_cls[common_rust_train_index[i][1]]
    Y_train = np.append(Y_train, train, axis=0)
    Y_test = np.append(Y_test, test, axis=0)

    train, test = gray_leaf[gray_leaf_train_index[i][0], :], gray_leaf[gray_leaf_train_index[i][1], :]
    X_train = np.append(X_train, train, axis=0)
    X_test = np.append(X_test, test, axis=0)
    train, test = gray_leaf_cls[gray_leaf_train_index[i][0]], gray_leaf_cls[gray_leaf_train_index[i][1]]
    Y_train = np.append(Y_train, train, axis=0)
    Y_test = np.append(Y_test, test, axis=0)

    train, test = healthy[healthy_train_index[i][0], :], healthy[healthy_train_index[i][1], :]
    X_train = np.append(X_train, train, axis=0)
    X_test = np.append(X_test, test, axis=0)
    train, test = healthy_cls[healthy_train_index[i][0]], healthy_cls[healthy_train_index[i][1]]
    Y_train = np.append(Y_train, train, axis=0)
    Y_test = np.append(Y_test, test, axis=0)

    X_train=X_train/255
    X_test=X_test/255

    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)

    vgg16_model = VGG16(input_shape = (224, 224, 1),
                                include_top = False,
                                weights = None)


    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_model.summary()

    '''model.add(Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

    model.fit(X_train,Y_train,epochs=60)

    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred,axis=1)
    acc = accuracy_score(Y_test, Y_pred)
    print('testing accuracy: ',acc)
    print('i: ',i)
    fo.write(str(i) + ',' + str(acc) + '\n')'''

'''fo.close()
model.summary()'''
'''acc = avg_acc / n_folds
print('Final Accuracy: ',acc)'''

'''cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
plt.show()
'''