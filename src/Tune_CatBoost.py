from catboost import CatBoostClassifier,CatBoostRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_catboost(trial,X,y,
                       eval_metric_model,scoring_metric,
                       n_trials,direction,problem_type,
                       N_folds,objective_func,stratify,gpu,repeat_n):



    param = {
         #"task_type":"GPU",
         #"devices":'0:1',
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.9),
        'one_hot_max_size':trial.suggest_int('one_hot_max_size', 1, 20),
        "nan_mode": trial.suggest_categorical("nan_mode", ["Min", "Max"]), 
        "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]), 
        'l2_leaf_reg':trial.suggest_int("l2_leaf_reg", 1, 30),
        'n_estimators':trial.suggest_int("n_estimators",10, 1500),
        #'sampling_frequency':trial.suggest_categorical("sampling_frequency", ["PerTree","PerTreeLevel"]),
        'grow_policy':trial.suggest_categorical("grow_policy", ['SymmetricTree', 'Lossguide', 'Depthwise']),
        'rsm' : trial.suggest_float('rsm', 0.01, 1.0),
        #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        
        "depth": trial.suggest_int("depth", 1, 16),

        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),        
        "used_ram_limit": "3gb",
        "eval_metric": eval_metric_model
    }
    
    if gpu==True:
        param["task_type"]="GPU"
        param["devices"]="0:1"
        
    if param["grow_policy"] == "Lossguide" or param["grow_policy"] == "Depthwise":
        param["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)
    
    if param["grow_policy"] == "Lossguide":
        param["max_leaves"] = trial.suggest_int("max_leaves", 1, 64)
       
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)


    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
    if problem_type=='classification':
        if objective_func==None:
            param["objective"]: trial.suggest_categorical("objective", ['Logloss','CrossEntropy'])
            param['auto_class_weights']=trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced"])
        else:
            param["objective"]: trial.suggest_categorical("objective", [objective_func])
            param['auto_class_weights']=trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced"])  
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(CatBoostClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    else:
        if objective_func==None:
            param["objective"]: trial.suggest_categorical("objective", ['RMSE','MAE'])
            if param["objective"] in ['Quantile', 'MAE', 'MAPE','LogCosh']:
                param["leaf_estimation_method"]: trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient","Exact"]) 
                
      
        else:
            param["objective"]: trial.suggest_categorical("objective", [objective_func])
            if param["objective"] in ['Quantile', 'MAE', 'MAPE','LogCosh']:
                param["leaf_estimation_method"]: trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient","Exact"])
        
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(CatBoostRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
        
    return np.mean(temp)
    
        
        
def CatBoost_tuner(X,y,
                   eval_metric_model,scoring_metric,
                   n_trials,direction,problem_type,N_folds=3,
                   objective_func=None,stratify=False,gpu=False,repeat_n=1):


    func = lambda trial: objective_catboost(trial,
                                            X,y,eval_metric_model,scoring_metric,
                                            n_trials,direction,problem_type,
                                            N_folds,objective_func,stratify,gpu,repeat_n)
    global study_catboost
    study_catboost = optuna.create_study(direction=direction)
    study_catboost.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_catboost.trials))
    print("Best trial:")
    trial = study_catboost.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    if problem_type=='classification':
        return CatBoostClassifier(**study_catboost.best_params)
    else:
        return CatBoostRegressor(**study_catboost.best_params)
        
        
"""
CatBoost_tuner(X=X2,y=y2,
           eval_metric_model='F1',
          scoring_metric="roc_auc",
          n_trials=10,
          N_folds=2,
          direction="maximize",
          stratify=True,
        problem_type='classification',
               
         )
"""