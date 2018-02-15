import xgboost as xgb

#from xgboost.sklearn import XGBClassifier

#dtrain = xgb.DMatrix('train.svm.txt')



xgb1 = xgb.sklearn.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#xgb.modelfit(xgb1, train, predictors)

