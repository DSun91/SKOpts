import optuna
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np

def ExtraTrees_objective(trial,X,y,problem_type,scoring_metric,N_folds,stratify,repeat_n):

    min_samples_split = trial.suggest_int('min_samples_split', 2, 12)
    min_impurity_decrease= trial.suggest_float('min_impurity_decrease',0.0, 0.01)
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 25)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features=trial.suggest_uniform('max_features', 0.1, 1.0)
    min_weight_fraction_leaf=trial.suggest_uniform('min_weight_fraction_leaf', 0.00001,0.4)
    bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
    class_weight= trial.suggest_categorical('class_weight',["balanced","balanced_subsample"])
    
    if problem_type=='classification':
        model = ExtraTreesClassifier(n_estimators = n_estimators,
                                     max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     criterion = criterion,
                                     class_weight=class_weight,
                                     min_impurity_decrease=min_impurity_decrease,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     bootstrap=bootstrap,
                                     random_state = 0
                                     )
    else:
        model = ExtraTreesRegressor(n_estimators = n_estimators,
                                     max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     criterion = criterion,
                                     #class_weight=class_weight,
                                     min_impurity_decrease=min_impurity_decrease,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     bootstrap=bootstrap,
                                     random_state = 0
                                     ) 
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)
    temp=[]
    for i in range(repeat_n):
        scores=cross_val_score(model,X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
        temp.append(np.mean(scores))
    
    return np.mean(temp)


    
def ExtraTrees_tuner(X,y,problem_type,scoring_metric,n_trials,direction,N_folds=3,stratify=False,repeat_n=1):
   
    func = lambda trial: ExtraTrees_objective(trial,X,y,problem_type,scoring_metric,n_trials,direction,N_folds,stratify,repeat_n)
    
    study_ExtraTrees = optuna.create_study(direction=direction)
    study_ExtraTrees.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_ExtraTrees.trials))
    print("Best trial:")
    trial = study_ExtraTrees.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return ExtraTreesClassifier(**study_ExtraTrees.best_params)
    else:
        return ExtraTreesRegressor(**study_ExtraTrees.best_params)