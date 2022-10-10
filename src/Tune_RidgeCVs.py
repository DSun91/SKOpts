from sklearn.linear_model import RidgeClassifierCV,RidgeCV
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np


def objective_RidgeCV(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    a=trial.suggest_float("a",0.0, 0.1)
    b=trial.suggest_float("b",0.1, 1)
    c=trial.suggest_float("c",1, 100)
    param = {}
    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
    if problem_type=='classification':
      
        param["alphas"]=(a,b,c)
        param["class_weight"]=trial.suggest_categorical("class_weight",["balanced",None])
        for i in range(repeat_n):
            scores=cross_val_score(RidgeClassifierCV(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores)) 
        
    else:
        param["alphas"]=(a,b,c)
        param["gcv_mode"]=trial.suggest_categorical("gcv_mode",["auto", "svd","eigen"])
        param["alpha_per_target"]=trial.suggest_categorical("alpha_per_target",[True,False])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(RidgeCV(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores)) 
        
        
    
    return np.mean(temp)
    


def RidgeCVs_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type='classification',repeat_n=1):
 
    func = lambda trial: objective_RidgeCV(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_RidgesCV
    study_RidgesCV = optuna.create_study(direction=direction)
    study_RidgesCV.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_RidgesCV.trials))
    print("Best trial:")
    trial = study_RidgesCV.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    if problem_type=='classification':
        return RidgeClassifierCV(**study_RidgesCV.best_params)
    else:
        return RidgeCV(**study_RidgesCV.best_params)