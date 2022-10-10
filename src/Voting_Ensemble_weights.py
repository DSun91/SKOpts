from itertools import product
from sklearn.ensemble import VotingClassifier,VotingRegressor
import optuna
from itertools import combinations
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np

def objective_Voting_weights(trial,X,y,
                     scoring_metric,
                     models_tps,
                     voting_type,
                     n_trials,
                     problem_type,
                     N_folds,
                     stratify,
                     weights_l):

    param={'weights':trial.suggest_categorical('weights', weights_l)}
    #param['weights']=trial.suggest_categorical('weights', weights_sys)
       
    #print(param['weights'])
    
    if stratify==True:
        cv=StratifiedKFold(n_splits=N_folds,shuffle=True)
    else:
        cv=KFold(n_splits=N_folds,shuffle=True)


    if problem_type=='classification':
        scores=cross_val_score(VotingClassifier(estimators=models_tps,voting=voting_type,weights=param['weights']),
                               X,y,scoring=scoring_metric, error_score="raise",cv=cv, n_jobs=-1)
        
    else:
        scores=cross_val_score(VotingRegressor(estimators=models_tps,weights=param['weights']),X,y,error_score="raise",
                               scoring=scoring_metric, cv=cv, n_jobs=-1)
        
    return np.mean(scores)
    
def stack_best_weights(X,y,
                scoring_metric,
                n_trials,
                direction,
                problem_type,
                models_list,
                N_folds=3,
                stratify=False,
                voting_type=None,
                weights_list=[1,2,3]):

    if len(models_list)>0:
        models_tps=[(x,y) for x,y in zip(['C'+str(i) for i in range(len(models_list))],models_list)]
    
    p = product(weights_list,repeat=len(models_tps))
    weights_l=[list(x) for x in p]    
    func = lambda trial: objective_Voting_weights(trial,X,y,
                                            scoring_metric,
                                            models_tps,
                                            voting_type,
                                            n_trials,
                                            problem_type,
                                            N_folds,
                                            stratify,
                                            weights_l)
    
    global study_voting_weights
    study_voting_weights = optuna.create_study(direction=direction)
    study_voting_weights.optimize(func, n_trials=n_trials)

    print("Number of finished trials: ", len(study_voting_weights.trials))
    print("Best trial:")
    trial = study_voting_weights.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
 
    for key, value in trial.params.items():
        
        print("    {}: {}".format(key, value))
        
    if problem_type=='classification':
        return VotingClassifier(estimators=models_list,voting=voting_type,weights=study_voting_weights.best_params),study_voting_weights
    else:
        return VotingRegressor(estimators=models_list,weights=study_voting_weights.best_params),study_voting_weights
        

stack_best_weights(X=X2,y=y2,
          scoring_metric="roc_auc",
          n_trials=10,
          N_folds=5,
          direction="maximize",
          stratify=True,
          problem_type='classification',
          voting_type='soft',
          models_list=classifiers
          weights_list=[0,1,2,3,4,5]
         )