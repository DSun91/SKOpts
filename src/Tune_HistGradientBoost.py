from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_HistG(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    
    param = {
     
        
        "learning_rate":trial.suggest_float("learning_rate", 0.00001, 1.0),
        "max_iter":trial.suggest_int("max_iter", 100, 300),
        "max_depth":trial.suggest_int("max_depth", 3, 12),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 10, 50),
        "max_leaf_nodes":trial.suggest_int("max_leaf_nodes", 3, 100),
        "max_bins":trial.suggest_int("max_bins", 10, 255),
        #"monotonic_cst":trial.suggest_categorical("monotonic_cst", [-1,0,1])
         
    }
    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
    if problem_type=='classification':
        
        param["loss"]=trial.suggest_categorical("loss", ["log_loss", "auto", "binary_crossentropy"])

        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(HistGradientBoostingClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
    
    
    else:
        param["loss"]=trial.suggest_categorical("loss", ['squared_error', 'absolute_error', 'poisson'])
        cv=KFold(n_splits=N_folds,shuffle=True)
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(HistGradientBoostingRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    return np.mean(temp)
    


def HistGradientBoost_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type='classification',repeat_n=1):
 
    func = lambda trial: objective_HistG(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_HistG
    study_HistG = optuna.create_study(direction=direction)
    study_HistG.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_HistG.trials))
    print("Best trial:")
    trial = study_HistG.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return HistGradientBoostingClassifier(**study_HistG.best_params)
    else:
        return HistGradientBoostingRegressor(**study_HistG.best_params)