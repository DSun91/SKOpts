# SkOpts
Library providing high-level functions for easy hyperparameters tuning of models using Optuna.<br /><br />
This repository contains the project of a library that offers ready-to-use functions to optimize the hyperparameters of some of models provided by the Sklearn API, XGBoost, LightGBM and CatBoost.

Installation command:
```
!pip install install SkOpts
```


Code snippet to optimize a classification model:

```
from SkOpts import SKOptuna

LGBM_tuned=SKOptuna.LGBM_tuner(X=Xtrain,y=y_train,
                    scoring_metric='roc_auc',
                    n_trials=100,
                    N_folds=5,
                    direction='maximize',
                    stratify=True,
                    problem_type='classification')
```
Code snippet to optimize regression model:

```
from SkOpts import SKOptuna

XGB_tuned=SKOptuna.XGB_tuner(X=Xtrain,y=y_train,
                    scoring_metric='neg_mean_squared_error',
                    n_trials=100,
                    N_folds=5,
                    direction='minimize',
                    problem_type='regression')
```

| Parameter  | Usage|
| ------------- | ------------- |
| **"X"**| Training dataset without target variable.|
|**"y"**| Target variable.|
| **"scoring_metric"**  | Metric to optimize.<br/>  |
| **"n_trials"** | Number of trials to execute optimization.<br/>  |
| **"N_folds"** | Number of folds for cross validation.<br/> |
| **"direction"**  | Equals "maximize" or "minimize".<br/>  |
| **"problem_type"**  | Equals "classification" or "regression".<br/>  |
| **"stratify"** | Stratify cv splits based on target distribuition [True or False]<br/>  |

The **'scoring_metric'** parameter takes the same values from sklearn API (link of available list: https://scikit-learn.org/stable/modules/model_evaluation.html) 
