import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D , Flatten, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy
from sklearn.metrics import accuracy_score,confusion_matrix

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

n_folds = 10
cv = KFold(n_splits=n_folds, shuffle=True)

blight_train_index = []
common_rust_train_index = []
gray_leaf_train_index = []
healthy_train_index = []

for train_index, test_index in cv.split(blight):
    blight_train_index.append([train_index, test_index])

for train_index, test_index in cv.split(common_rust):
    common_rust_train_index.append([train_index, test_index])

for train_index, test_index in cv.split(gray_leaf):
    gray_leaf_train_index.append([train_index, test_index])

for train_index, test_index in cv.split(healthy):
    healthy_train_index.append([train_index, test_index])

toPath = 'C:/Users/Raj/MyProg/Corn_Disease/codes/train_index/'
train_pickle = ['blight','common_rust','gray_leaf','healthy']

bl = toPath+train_pickle[0]+'_train_index_2'+".pkl"
fo = open(bl, "wb")
pickle.dump(blight_train_index,fo)
fo.close()

cr = toPath+train_pickle[1]+'_train_index_2'+".pkl"
fo = open(cr, "wb")
pickle.dump(common_rust_train_index,fo)
fo.close()

gl = toPath+train_pickle[2]+'_train_index_2'+".pkl"
fo = open(gl, "wb")
pickle.dump(gray_leaf_train_index,fo)
fo.close()

hl = toPath+train_pickle[3]+'_train_index_2'+".pkl"
fo = open(hl, "wb")
pickle.dump(healthy_train_index,fo)
fo.close()