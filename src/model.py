import lightgbm as lgb
import gc


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='label', objective='binary', metrics='binary_logloss',
                      feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    # parameters: https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.001,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 63,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # Minimum number of data need in a child(min_data_in_leaf)
        'min_child_samples': 5,
        'max_bin': 65535,  # Number of bucketed bin for feature values
        'subsample': 0.9,  # Subsample ratio of the training instance.
        # Subsample ratio of columns when constructing each tree.
        'colsample_bytree': 0.8,
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.1,  # L1 regularization term on weights
        'reg_lambda': 0.1,  # L2 regularization term on weights
        'nthread': 16,
        'verbose': 1,
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
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
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics + ":", evals_results['valid']
          [metrics][bst1.best_iteration - 1])

    return (bst1, bst1.best_iteration)


def train_model(df, predictors):
    params = {
        'scale_pos_weight': 5
    }

    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              df[:1200],
                                              df[1200:],
                                              predictors,
                                              objective='binary',
                                              metrics='binary_logloss',
                                              early_stopping_rounds=30,
                                              verbose_eval=True,
                                              num_boost_round=10000)
    return bst, best_iteration
