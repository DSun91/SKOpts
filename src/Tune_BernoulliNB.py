from sklearn.naive_bayes import BernoulliNB
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np

def objective_BernoulliNB(trial,X,y, scoring_metric,N_folds,stratifyrepeat_n):
    #(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    
    param = {
     
        
        "alpha":trial.suggest_float("alpha", 0.0, 10000.0),
        "binarize":trial.suggest_float("binarize", 0.0, 1.0),
        "fit_prior":trial.suggest_categorical("fit_prior",[True,False]),
         
    }
    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
    
    temp=[] 
    for i in range(repeat_n):
        scores=cross_val_score(BernoulliNB(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
        temp.append(np.mean(scores))
    
    return np.mean(temp)
    


def BernoulliNB_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,repeat_n=1):
 
    func = lambda trial: objective_BernoulliNB(trial,X,y, scoring_metric,N_folds,stratify,repeat_n)
    global study_BernoulliNB
    study_BernoulliNB = optuna.create_study(direction=direction)
    study_BernoulliNB.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_BernoulliNB.trials))
    print("Best trial:")
    trial = study_BernoulliNB.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return BernoulliNB(**study_BernoulliNB.best_params)