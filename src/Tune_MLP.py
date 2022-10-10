from sklearn.neural_network import MLPClassifier,MLPRegressor
import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
#trial.suggest_float(),
#trial.suggest_int()
#trial.suggest_categorical()


def objective_MLP(trial,X,y,scoring_metric,N_folds,stratify,problem_type,hidden_layer_sizes,repeat_n):
    
    
    param = {
        "hidden_layer_sizes":hidden_layer_sizes,
        "activation":trial.suggest_categorical("activation",['relu','identity','tanh','logistic']),
        "solver":trial.suggest_categorical("solver",['lbfgs', 'sgd', 'adam']),
        "alpha":trial.suggest_float("alpha",1e-6, 1e-2),
        #"learning_rate":trial.suggest_float("learning_rate",0.0001,1.0),
        "max_iter":trial.suggest_int("max_iter", 100, 400, step=1),
        "early_stopping":True,
        }


    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)

        
    if param['solver']=='sgd':
        param['learning_rate']=trial.suggest_categorical("learning_rate",['constant','invscaling','adaptive'])
        param['power_t']=trial.suggest_float("power_t",0.1, 0.9)
        param['momentum']=trial.suggest_float("momentum",0.0, 1.0)
        if param['momentum']>0:
            param['nesterovs_momentum']=trial.suggest_categorical("nesterovs_momentum",[True,False])
            
    if param['solver']=='adam':
        param['learning_rate_init']=trial.suggest_float("learning_rate_init",0.0001,0.1)
        param['beta_1']=trial.suggest_float("beta_1",0.0, 1)
        param['beta_2']=trial.suggest_float("beta_2",0.0, 1)
        param['epsilon']=trial.suggest_float("epsilon",1e-10,1e-7)
        
        
        
    if problem_type=='classification':  
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(MLPClassifier(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
    else:
        temp=[]
        for i in range(repeat_n):
            scores=cross_val_score(MLPRegressor(**param),X,y, scoring=scoring_metric, cv=cv, n_jobs=-1)
            temp.append(np.mean(scores))
    return np.mean(temp)



def MLP_tuner(X,y,scoring_metric,n_trials,N_folds=3,direction=None,stratify=False,problem_type=None,hidden_layer_sizes=(100,),repeat_n=1):
 
    func = lambda trial: objective_MLP(trial,X,y,scoring_metric,N_folds,stratify,problem_type,hidden_layer_sizes,repeat_n)
    global study_MLP
    study_MLP = optuna.create_study(direction=direction)
    study_MLP.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_MLP.trials))
    print("Best trial:")
    trial = study_MLP.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    if problem_type=='classification':
        return MLPClassifier(**study_MLP.best_params)
    else:
        return MLPRegressor(**study_MLP.best_params)