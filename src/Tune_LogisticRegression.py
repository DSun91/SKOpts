from sklearn.linear_model import LogisticRegression
import optuna
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np


def objective_LogisticRegression(trial,X,y, scoring_metric,N_folds,stratify,repeat_n):
    
    
    param = {
      
        #"penalty":trial.suggest_categorical("penalty", ['l1','l2','none']),
        #"tol": trial.suggest_float("tol", 1e-5, 1e-3),
        "C": trial.suggest_float("C", 1e-5, 1e5),
        "solver":trial.suggest_categorical("solver", ['newton-cg','lbfgs','liblinear','sag','saga']),
        "class_weight":trial.suggest_categorical("class_weight", ['balanced',None]),
        "max_iter": trial.suggest_int("max_iter", 10, 1500)
        
    }


            
    if param["solver"]=='liblinear':
        param["penalty"]:trial.suggest_categorical("penalty", ['l2','l1'])
    
    if param["solver"]=='newton-cg' or param["solver"]=='lbfgs' or param["solver"]=='sag':
        param["penalty"]:trial.suggest_categorical("penalty", ['l2','none'])
        
    
    if param["solver"]=='saga':
        param["penalty"]=trial.suggest_categorical("penalty",['elasticnet', 'l1', 'l2', 'none'])
        param["l1_ratio"]=trial.suggest_float("l1_ratio", 0,1)
        

        
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
                                      
    temp=[]
    for i in range(repeat_n):
        scores=cross_val_score(LogisticRegression(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
        temp.append(np.mean(scores))
    
   
    return np.mean(temp)



def LogisticRegression_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,repeat_n=1):
 
    func = lambda trial: objective_LogisticRegression(trial,X,y, scoring_metric,N_folds,stratify)
    global study_LogisticRegression
    study_LogisticRegression = optuna.create_study(direction=direction)
    study_LogisticRegression.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_LogisticRegression.trials))
    print("Best trial:")
    trial = study_LogisticRegression.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    return LogisticRegression(**study_LogisticRegression.best_params)
