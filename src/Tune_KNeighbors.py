from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
#trial.suggest_float(),
#trial.suggest_int()
#trial.suggest_categorical()


def objective_KNeighbors(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    

    param = {
        "n_jobs":-1,
        "n_neighbors":trial.suggest_int("n_neighbors", 1, 50, step=1),
        "weights":trial.suggest_categorical("weights",['uniform','distance']),
        "algorithm":trial.suggest_categorical("algorithm",['auto','ball_tree','kd_tree','brute']),
        "leaf_size":trial.suggest_int("leaf_size", 1, 100),
        "p":trial.suggest_float("p", 1, 3),
        }


    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)

    if problem_type=='classification':  
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(KNeighborsClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
    else:
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(KNeighborsRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
    return np.mean(temp)



def KNeighbors_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type=None,repeat_n=1):
 
    func = lambda trial: objective_KNeighbors(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_KNeighbors
    study_KNeighbors = optuna.create_study(direction=direction)
    study_KNeighbors.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_KNeighbors.trials))
    print("Best trial:")
    trial = study_KNeighbors.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return KNeighborsClassifier(**study_KNeighbors.best_params)
    else:
        return KNeighborsRegressor(**study_KNeighbors.best_params)