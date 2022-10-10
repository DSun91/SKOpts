class SKOptuna:
    def __init__(self) -> None:
        pass

    from Tune_Bagging import Bagging_tuner   
    from Tune_BernoulliNB import BernoulliNB_tuner
    from Tune_CatBoost import CatBoost_tuner
    from Tune_DecisionTree import DecisionTree_tuner
    from Tune_Extratrees import ExtraTrees_tuner
    from Tune_GaussianNB import GaussianNB_tuner
    from Tune_HistGradientBoost import HistGradientBoost_tuner
    from Tune_KNeighbors import KNeighbors_tuner
    from Tune_Lgbm import LGBM_tuner
    from Tune_MLP import MLP_tuner
    from Tune_RandomForest import RandomForest_tuner
    from Tune_Ridge import Ridge_tuner
    from Tune_RidgeCVs import RidgeCVs_tuner
    from Tune_Xgboost import XGB_tuner
    from Tune_LogisticRegression import LogisticRegression_tuner
    