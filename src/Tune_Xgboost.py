from xgboost import XGBClassifier,XGBRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_xgb(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    
    
    param = {
        "verbosity": 0,
        #"objective": xgb_objective,#"binary:logistic"
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "max_depth":trial.suggest_int("max_depth", 3, 12, step=2),
        "min_child_weight" : trial.suggest_int("min_child_weight", 2, 10000),
        "max_delta_step" : trial.suggest_int("max_delta_step", 0, 10000),
         "eta" : trial.suggest_float("eta", 0.001, 1.0, log=True),
        "gamma":trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        #"sampling_method":trial.suggest_categorical("sampling_method", ['uniform',]),
        "max_bin":trial.suggest_int("max_bin",100, 10000),
        
        
    }


    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
        
    if problem_type=='classification':
        param['objective']=trial.suggest_categorical("objective", ['binary:logistic','binary:logitraw','binary:hinge'])                                     
        param['scale_pos_weight']=trial.suggest_float("scale_pos_weight", 1.0, 14, log=True)
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(XGBClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    else:
        
        param['objective']=trial.suggest_categorical("objective", ['reg:squarederror','reg:squaredlogerror'])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(XGBRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    return np.mean(temp)



def XGB_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type='classification',repeat_n=1):
 
    func = lambda trial: objective_xgb(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_xgb
    study_xgb = optuna.create_study(direction=direction)
    study_xgb.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_xgb.trials))
    print("Best trial:")
    trial = study_xgb.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return XGBClassifier(**study_xgb.best_params)
    else:
        return XGBRegressor(**study_xgb.best_params)