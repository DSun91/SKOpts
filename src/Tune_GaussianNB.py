from sklearn.naive_bayes import GaussianNB
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_GaussianNB(trial,X,y, scoring_metric,N_folds,stratify,repeat_n):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    
    param = {
        "var_smoothing":trial.suggest_float("var_smoothing", 1e-12, 1e-6),
    }
    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)

    temp=[]  
    for i in range(repeat_n):
        scores=cross_val_score(GaussianNB(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
        temp.append(np.mean(scores))
    
    return np.mean(temp)
    


def GaussianNB_tuner(X,y,scoring_metric,n_trials,direction,N_folds=3,stratify=False,repeat_n=1):
 
    func = lambda trial: objective_GaussianNB(trial,X,y, scoring_metric,N_folds,stratify,repeat_n)
    global study_GaussianNB
    study_GaussianNB = optuna.create_study(direction=direction)
    study_GaussianNB.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_GaussianNB.trials))
    print("Best trial:")
    trial = study_GaussianNB.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return GaussianNB(**study_GaussianNB.best_params)