import lightgbm as lgb
import gc
from common import *


def lgb_model(dtrain, dvalid, predictors, target, objective, metrics):

    # parameters: https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    lgb_params = {
        'objective': objective,
        'metric': metrics,
        'scale_pos_weight': 5,
        'boosting_type': 'gbdt',
        'learning_rate': 0.001,
        'num_leaves': 63,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # Minimum number of data need in a child(min_data_in_leaf)
        'min_child_samples': 2,
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.9,  # Subsample ratio of the training instance.
        'subsample_freq': 10,  # frequence of subsample, <=0 means no enable
        # Subsample ratio of columns when constructing each tree.
        'colsample_bytree': 0.8,
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.5,  # L1 regularization term on weights
        'reg_lambda': 0.5,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 1,
    }

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=None
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=None
                          )
    del dtrain
    del dvalid
    gc.collect()

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgvalid],
                     valid_names=['valid'],
                     evals_result=evals_results,
                     num_boost_round=10000,
                     early_stopping_rounds=30,
                     verbose_eval=10,
                     feval=None)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics + ":", evals_results['valid']
          [metrics][bst1.best_iteration - 1])

    # feature names
    print('Feature names:', bst1.feature_name())

    # feature importances
    print('Feature importances:', list(bst1.feature_importance()))

    return (bst1, bst1.best_iteration)


def train_model(df, predictors, target, num_train):
    objective = 'binary'
    metrics = 'binary_logloss'

    print('len of df:', len(df))

    (bst, best_iteration) = lgb_model(df[:num_train], df[
        num_train:], predictors, target, objective, metrics)
    return bst, best_iteration


def load_model(model_file):
    return lgb.Booster(model_file=model_file)
