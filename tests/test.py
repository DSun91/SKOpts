from SKOpts import SKOptuna


import pandas as pd
import pickle
import sys  
sys.path.insert(0, 'C:/Users/dials/Desktop/DSunpy/')
from itertools import combinations
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
pd.set_option('display.expand_frame_repr', False)
from sklearn.preprocessing import PowerTransformer
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from catboost import cv
import numpy as np
import numpy as np
import optuna
from xgboost import XGBClassifier
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split#For splitting
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, Pool 
import warnings
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
def optimizedata(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('str')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_train=df_train.drop(['Unnamed: 0'],axis=1)
df_test=df_test.drop(['Unnamed: 0'],axis=1)#'encounter_id','patient_id','patient_id','hospital_id'
optimizedata(df_train)
optimizedata(df_test)
df_train=df_train.drop(['Unnamed: 83'],axis=1)
df_test=df_test.drop(['Unnamed: 83'],axis=1)
X1, y1=df_train.drop(['hospital_death'],axis=1),df_train['hospital_death']
for i in df_test:
    if df_train[i].dtypes=='object':
        df_train[i]=df_train[i].fillna('Unknown')
        df_test[i]=df_test[i].fillna('Unknown')
    else:
        df_train[i]=df_train[i].fillna(-10)
        df_test[i]=df_test[i].fillna(-10)
categorical_features_indices =np.where((X1.dtypes != np.float32) & (X1.dtypes != np.float16))[0]
categorical_features_indices=np.delete(categorical_features_indices, 0)
for i in categorical_features_indices:
    df_train.iloc[:,i]=df_train.iloc[:,i].astype('category')
    df_test.iloc[:,i]=df_test.iloc[:,i].astype('category')
df_train['hospital_death'].value_counts()
X1, y1=df_train.drop(['hospital_death'],axis=1),df_train['hospital_death']     
df_train2=df_train.copy()
df_test2=df_test.copy()
df_train2['type']=1
df_test2['type']=2
df_complete=pd.concat([df_train2,df_test2],axis=0)

for i in df_test2.columns:
    #print(df_complete[i].dtypes[0])
    if df_complete[i].dtypes=='category':
        #print(df_complete[i])
        df_complete[i]=LabelEncoder().fit_transform(df_complete[i])
      
    
df_train2=df_complete[df_complete['type']==1]
df_test2=df_complete[df_complete['type']==2]
df_train2=df_train2.drop(['type'],axis=1)
df_test2=df_test2.drop(['hospital_death','type'],axis=1)
X2, y2=df_train2.drop(['hospital_death'],axis=1),df_train2['hospital_death']

#(self,X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type=None)

BG=SKOptuna.XGB_tuner(X=X2,y=y2,scoring_metric='roc_auc',n_trials=15,N_folds=3,direction='maximize',stratify=False,
#problem_type='classification'
)
print(BG)