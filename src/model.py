import lightgbm as lgb
import gc


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='label', objective='binary', metrics='binary_logloss',
                      feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    # parameters: https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    lgb_params = {
        'objective': objective,
        'metric': metrics,
        'boosting_type': 'gbdt',
        'learning_rate': 0.001,
        'num_leaves': 63,  # we should let it be smaller than 2^(max_depth)
        'max_bin': 65535,  # Number of bucketed bin for feature values
        'min_child_samples': 5,
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

    print('len of df:', len(df))
    df = df.sample(frac=1).reset_index(drop=True)
    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              df[:20000],
                                              df[20000:],
                                              predictors,
                                              objective='binary',
                                              metrics='binary_logloss',
                                              early_stopping_rounds=30,
                                              verbose_eval=True,
                                              num_boost_round=10000)
    return bst, best_iteration
