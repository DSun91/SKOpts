from sklearn.ensemble import BaggingClassifier,BaggingRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import numpy as np



def objective_Bagging(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    
    
    param = {
        "n_estimators":trial.suggest_int("n_estimators", 1, 500, step=1),
        "max_features":trial.suggest_float("max_features",0.1,1.0),
        }


    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)

    if problem_type=='classification':
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(BaggingClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
       
    else:
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(BaggingRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
  
        
    return np.mean(temp)



def Bagging_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type=None,repeat_n=1):
 
    func = lambda trial: objective_Bagging(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_Bagging
    study_Bagging = optuna.create_study(direction=direction)
    study_Bagging.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_Bagging.trials))
    print("Best trial:")
    trial = study_Bagging.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return BaggingClassifier(**study_Bagging.best_params)
    else:
        return BaggingRegressor(**study_Bagging.best_params)