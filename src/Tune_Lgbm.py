from lightgbm import LGBMClassifier,LGBMRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
def objective_lgbm(trial,X,y,scoring_metric,N_folds,metric_model,stratify,problem_type,repeat_n):
   
    #train_x, test_x, train_y, test_y = train_test_split(X1, y1, test_size=0.25,stratify=y1,shuffle=True)
    #dtrain = lgb.Dataset(train_x, label=train_y)
 
    param= {
        'boosting_type':trial.suggest_categorical('boosting_type',['dart','gbdt','rf']),
        #'objective': "binary",
        'metric': metric_model,
        'n_estimators ': trial.suggest_int('n_estimators ', 1, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        #'is_unbalance':trial.suggest_categorical('is_unbalance',['true','false']), 
        'max_bin': trial.suggest_int('max_bin', 2, 2000),
        #'scale_pos_weight':trial.suggest_uniform('scale_pos_weight', 1, 40),
    }
   
 
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
    
    if problem_type=='classification':
        param['objective']=trial.suggest_categorical('objective',['binary','cross_entropy','cross_entropy_lambda'])
        param['is_unbalance']=trial.suggest_categorical('is_unbalance',['true','false'])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(LGBMClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    else:
        param['objective']=trial.suggest_categorical('objective',['rmse','mae','mape'])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(LGBMRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    return np.mean(scores)


def LGBM_tuner(X,y,scoring_metric,n_trials,N_folds=3,metric_model="",direction=None,stratify=False,problem_type='classification',repeat_n=1):
    

        
    func = lambda trial: objective_lgbm(trial,X,y,scoring_metric,N_folds,metric_model,stratify,problem_type,repeat_n)
    global study_lgbm
    study_lgbm = optuna.create_study(direction=direction)
    study_lgbm.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_lgbm.trials))
    print("Best trial:")
    trial = study_lgbm.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    if problem_type=='classification':
        return LGBMClassifier(**study_lgbm.best_params)
    else:
        return LGBMRegressor(**study_lgbm.best_params)