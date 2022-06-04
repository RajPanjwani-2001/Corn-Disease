import tensorflow as tf
from tensorflow.keras import Model
import pickle
import numpy as np
import pandas as pd
from config import config
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import *
from sklearn.preprocessing import MinMaxScaler

class ML :
    def factor_analysis(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        fa_obj = FactorAnalysis(n_components=n_components,random_state=0)
        fa_obj.fit(scaled_data)
        x = fa_obj.transform(scaled_data)
        return x


    def pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        pca_obj = PCA(n_components=n_components)
        pca_obj.fit(scaled_data)
        x_pca = pca_obj.transform(scaled_data)
        return x_pca

    def fast_ica(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        fi_obj = FastICA(n_components=n_components,random_state=0)
        fi_obj.fit(scaled_data)
        x = fi_obj.transform(scaled_data)
        return x
   
    def incremental_pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = IncrementalPCA(n_components=n_components)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x
        

    def kernel_pca(self,data,n_components,kernel='linear'): #kernel = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = KernelPCA(n_components=n_components,kernel=kernel)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x


    def lda(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = LatentDirichletAllocation(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def mini_batch_dict_learning(self,data,n_components,transform_algorithm= 'lasso_lars'): #transform_algorithm = 'lasso_lars','lasso_cd'
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = MiniBatchDictionaryLearning(n_components=n_components,transform_algorithm=transform_algorithm)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x
        
    def mini_batch_sparse_pca(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = MiniBatchSparsePCA(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def nmf(self,data,n_components,init='random'): #init{‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = NMF(n_components=n_components, init=init, random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

    def sparse_pca(self,data,n_components):
        '''scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)'''
        obj = SparsePCA(n_components=n_components,random_state=0)
        obj.fit(data)
        x = obj.transform(data)
        return x

    def truncated_svd(self,data,n_components):
        scaler = MinMaxScaler()
        scaler.fit(data)

        scaled_data = scaler.transform(data)
        obj = TruncatedSVD(n_components=n_components,random_state=0)
        obj.fit(scaled_data)
        x = obj.transform(scaled_data)
        return x

class FeatureSelection:
    def getInformationGain(self, X, Y):
        info = np.array(mutual_info_regression(X,Y))
        info = (info - np.min(info)) / (np.max(info) - np.min(info))
        info = np.round(info, decimals=6)
        return info

    def getKFeatures(self, K, function, trainX, trainY, testX, testY):
        functions_dct = {'chi2': chi2, 'mi': mutual_info_regression, 'anova': f_classif, 'pearson': f_regression}
        if function not in functions_dct:
            print('Wrong function name')
            raise Exception('Wrong function name : ' + function)

        selector = SelectKBest(functions_dct[function], k=K)
        selector.fit(trainX, trainY)
        trainX_new = selector.transform(trainX)
        testX_new = selector.transform(testX)
        return trainX_new, testX_new

    def getFeatureRanking(self, trainX, trainY, estimator):
        rfecv = RFECV(estimator=estimator, min_features_to_select=1, step=np.ceil(trainX.shape[1]/10), cv=3)
        rfecv.fit(trainX, trainY)
        ranks = rfecv.ranking_
        return ranks

    def getTopFeatures(self, trainX, trainY, testX, estimator):
        ranks = self.getFeatureRanking(trainX, trainY, estimator)
        indices = np.array(np.where(ranks == 1)).flatten()
        trainX_new = trainX[:, indices]
        testX_new = testX[:, indices]
        return trainX_new, testX_new

class TransferLearningFeatureSelection():
    def __init__(self, model_name, base_model, data_pickle_file, cls_pickle_file):
        self.__model_name = model_name
        self.__base_model = base_model
        self.__data_pickle_file = data_pickle_file
        self.__cls_pickle_file = cls_pickle_file

    def get_model(self):
        for layer in self.__base_model.layers:
            layer.trainable = False

        x = self.__base_model.output
        x = tf.keras.layers.Flatten()(x)
        model = Model(inputs=self.__base_model.input, outputs=x)
        return model

    def get_model_based_features(self):
        model = self.get_model()
        data = self.get_data()
        model_features = model.predict(data)
        print('Model Features Shape : ', model_features.shape)
        return model_features

    def create_model_features_pickle(self):
        model_features = self.get_model_based_features()
        fp = open(self.__model_name + "_features.pickle", "wb")
        pickle.dump(model_features, fp)
        fp.close()

    def get_top_model_features(self):
        cls = self.get_cls_labels()

        #fp = open(self.__model_name + "_features.pickle", "rb")
        fp = open('Combined.pkl', "rb")
        model_features = pickle.load(fp)
        fp.close()

        fe = FeatureSelection()
        xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1)
        top_features, _ = fe.getTopFeatures(model_features, cls, model_features, estimator=xgb)
        return top_features

    def create_top_model_features_pickle(self):
        top_features = self.get_top_model_features()
        print(top_features.shape)
        fp = open(self.__model_name + "_selected_top_features.pickle", "wb")
        pickle.dump(top_features, fp)
        fp.close()

    def get_top_model_features_from_pickle(self):
        fp = open('Combined.pkl', "rb")
        #fp = open(self.__model_name + "_selected_top_features.pickle", "rb")
        top_features = pickle.load(fp)
        fp.close()
        return top_features

    def get_data(self):
        fp = open(self.__data_pickle_file, "rb")
        data = pickle.load(fp)
        fp.close()
        return data

    def get_cls_labels(self):
        fp = open(self.__cls_pickle_file, "rb")
        cls = pickle.load(fp)
        fp.close()
        return cls

class XGBoostClassifierTune():
    def __init__(self, X, Y, folds):
        # Converting the dataframe into XGBoost’s Dmatrix object
        self.__dtrain = xgb.DMatrix(X, label=Y)
        self.__X = X
        self.__Y = Y
        self.__folds = folds
    # Bayesian Optimization function for xgboost
    def tune_xgb(self):
        pass

        