from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
#trial.suggest_float(),
#trial.suggest_int()
#trial.suggest_categorical()


def objective_DecisionTree(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n):
    
    
    param = {
       
        "splitter":trial.suggest_categorical("splitter",['best','random']),
        "max_features":trial.suggest_categorical("max_features",['sqrt', 'log2', None]),
        "ccp_alpha":trial.suggest_float("ccp_alpha",0.0,0.01),
         "max_depth":trial.suggest_int("max_depth", 3, 12, step=1),
        "min_impurity_decrease":trial.suggest_float("min_impurity_decrease",0.0,0.01)
        }


    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)

    if problem_type=='classification':  
        param["criterion"]=trial.suggest_categorical("criterion",['gini', 'entropy', 'log_loss'])
        param['class_weight']=trial.suggest_categorical("class_weight", ['balanced',None])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(DecisionTreeClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
       
    else:
        param["criterion"]=trial.suggest_categorical("criterion",['squared_error', 'absolute_error', 'poisson','friedman_mse'])
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(DecisionTreeRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
        
    return np.mean(scores)



def DecisionTree_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type=None,repeat_n=1):
 
    func = lambda trial: objective_DecisionTree(trial,X,y, scoring_metric,N_folds,stratify,problem_type,repeat_n)
    global study_DecisionTree
    study_DecisionTree = optuna.create_study(direction=direction)
    study_DecisionTree.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_DecisionTree.trials))
    print("Best trial:")
    trial = study_DecisionTree.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return DecisionTreeClassifier(**study_DecisionTree.best_params)
    else:
        return DecisionTreeRegressor(**study_DecisionTree.best_params)