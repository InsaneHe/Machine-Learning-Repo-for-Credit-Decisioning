#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier #sklearn xgboost
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')
print(train.shape, test.shape)
target='Disbursed'
IDcol = 'ID'

#test_results = pd.read_csv('test_results.csv')
def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        # train
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # pred
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # eval
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    # ft-imp
    feat_imp = pd.Series(alg.feature_importances_, index=list(dtrain[predictors].columns)).sort_values(ascending=False)
    plt.figure(figsize=(16, 5))
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

