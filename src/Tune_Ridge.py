from sklearn.linear_model import RidgeClassifier,Ridge
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_Ridge(trial,X,y,scoring_metric,N_folds,stratify,problem_type,repeat_n):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
 
    param = {
        
        'alpha' : trial.suggest_float('alpha', 0.001, 10000),
        'solver':trial.suggest_categorical("solver",["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
    }
    if param["solver"]=="lbfgs":
            param["positive"]=trial.suggest_categorical("positive",[True,False])
    if param["solver"]=="sag" or param["solver"]=="saga":
        param["random_state"]=trial.suggest_int("random_state",0,50)
            
            
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
    if problem_type=='classification':
        param["class_weight"]=trial.suggest_categorical("class_weight",["balanced",None])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(RidgeClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
        
    else:    
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(Ridge(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))   
        
        
    
    return np.mean(temp)
    


def Ridge_tuner(X,y,scoring_metric,n_trials,direction,N_folds=3,stratify=False,problem_type='classification',repeat_n=1):
 
    func = lambda trial: objective_Ridge(trial,X,y,scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_Ridge
    study_Ridge = optuna.create_study(direction=direction)
    study_Ridge.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_Ridge.trials))
    print("Best trial:")
    trial = study_Ridge.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return RidgeClassifier(**study_Ridge.best_params)
    else:
        return Ridge(**study_Ridge.best_params)