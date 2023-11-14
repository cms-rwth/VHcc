from coffea.util import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mplhep as hep
import hist
import argparse, sys, os, arrow, glob, yaml
from matplotlib.offsetbox import AnchoredText
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedKFold
import json

#######################################################################################
## Create the folder to save the data if it doesn't exist and read in the dataframe ###
#######################################################################################
net_path = "/net/scratch_cms3a/vaulin/"
folder_save = 'eval_23_08_08'
roi = 'low_mumu'
if not os.path.exists(f"./plot/{folder_save}"):
    os.mkdir(f"./plot/{folder_save}")
if not os.path.exists(f"./plot/{folder_save}/ROI_simple"):
    os.mkdir(f"./plot/{folder_save}/ROI_simple")

if not os.path.exists(net_path + f"plot/{folder_save}"):
    os.mkdir(net_path + f"plot/{folder_save}")
df = pd.read_csv(f'./plot/{folder_save}/xgb_training_dataset_{roi}.csv')


bgs = ['DY', "ZZ", "WZ", "tt", "ZHtobb"]
bg_choice = 2
bg_choice_2 = 0
bg_choice_3 = 1

eta = 0.03
#eta = 0.03, 0.12, 0.3, 0.45, 0.8

df = df[(df.target_bg == 0)|(df.target_bg == bg_choice+1)|(df.target_bg == bg_choice_2+1)|(df.target_bg == bg_choice_3+1)] 

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)


########################################################################################
########## drop target from df and bring it to a separate column, drop weights #########
########################################################################################
X = df.drop("target", axis = 1)
X = X.drop("target_bg", axis = 1)
print(X)
X = X.drop(f"wei_{roi}", axis = 1)
X = X.drop(f"Z_mass_{roi}", axis = 1)
X = X.drop(f"Z_pt_gen_{roi}", axis = 1)
X = X.drop(f"Z_mass_gen_{roi}", axis = 1)
print(X)
print(X.info())

y = df["target"]
print(y)


########################################################################################
################# GRID search attempt ##################################################
########################################################################################
'''
from sklearn.model_selection import GridSearchCV

### Creat the parameter grid
gbm_param_grid = {'max_depth' : [3, 4, 5, 6, 7, 8, 9], 'min_child_weight' : [1], 'gamma' : [0], 'subsample' : [0.8], 'colsample_bytree' : [0.8], 'reg_alpha' : [0.005], 'n_estimators': [1000]}

gbm = xgb.XGBRegressor()

grid_mse = GridSearchCV(param_grid = gbm_param_grid, estimator = gbm, scoring = 'neg_mean_squared_error', cv = 4, verbose = 1)

grid_mse.fit(X,y)


print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))  
'''

########################################################################################
############# An attempt to do hyperparameter tuning for the classifier fit ############
########################################################################################
space = {"max_depth": hp.quniform("max_depth", 3, 18, 1),
         "gamma": hp.uniform("gamma", 1, 9),
         "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
         "reg_lambda": hp.uniform("reg_lambda", 0, 1),
         "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
         "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
         "n_estimators": 200,
         "learning_rate": hp.uniform("learning_rate", 0.001, 0.1),
         "subsample": hp.uniform("subsample", 0.8, 1),
         "seed":0}

#learning_rate = space['learning_rate'],

def objective(space):
    clf = xgb.XGBClassifier( n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'], reg_alpha = int(space['reg_alpha']), min_child_weight = int(space['min_child_weight']), colsample_bytree = int(space['colsample_bytree']), eval_metric = 'auc', early_stopping_rounds = 10)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    clf.fit(X_train, y_train, eval_set = evaluation, verbose = False)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print("SCORE: ", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK} 


#########################################################################################
############# Create pipelines for xgb training #########################################
#########################################################################################
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy = "most_frequent")), ("oh-encode", OneHotEncoder(handle_unknown = "ignore", sparse = False)),])

from sklearn.preprocessing import StandardScaler
numeric_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy = "mean")), ("scale", StandardScaler())])

cat_cols = X.select_dtypes(exclude = "number").columns
num_cols = X.select_dtypes(include = "number").columns

print(cat_cols)
print(num_cols)

from sklearn.compose import ColumnTransformer

full_processor = ColumnTransformer(transformers = [("numeric", numeric_pipeline, num_cols), ("categorical", categorical_pipeline, cat_cols),])



X_processed = full_processor.fit_transform(X)
y_processed = SimpleImputer(strategy = "most_frequent").fit_transform(y.values.reshape(-1,1))

#########################################################################################
############ split dataset into training and test #######################################
#########################################################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, stratify = y_processed, random_state = 1121218)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101, stratify = y)
print(X_train)
print(X_test)
print(y_train)

############################################################################################################
######### preparing the XGB classifiers in 20 x 5-folds cross validation using repeated k-fold #############
############################################################################################################
cv = RepeatedKFold(n_splits = 8, n_repeats = 20, random_state = 101)
folds = [(train, test) for train, test in cv.split(X_train, y_train)]
#print(folds)
metrics = ['auc', 'fpr', 'tpr', 'thresholds']
results = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
results_zero_train = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
results_weak_train = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}

params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': eta}
with open(net_path + f"plot/{folder_save}/results_first_{eta}.json", 'w') as outfile:
    json.dump(results, outfile)



dtest = xgb.DMatrix(X_test, label = y_test)
#print(dtest)
for train, test in tqdm(folds, total = len(folds)):
    print('train')
    dtrain = xgb.DMatrix(X_train[train,:],
             label = y_train[train])
    dval = xgb.DMatrix(X_train[test, :], label = y_train[test])
    model = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                      verbose_eval = 1, early_stopping_rounds = 10, num_boost_round = 200) #num_boost_round = 1000, 200 is optimal
    model_zero_train = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                      verbose_eval = 1, early_stopping_rounds = 10, num_boost_round = 0) #num_boost_round = 1000, 200 is optimal
    model_weak_train = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                      verbose_eval = 1, early_stopping_rounds = 10, num_boost_round = 20) #num_boost_round = 1000, 200 is optimal
    sets = [dtrain, dval, dtest]
    for i, ds in enumerate(results.keys()):
        print(i)
        y_preds = model.predict(sets[i])
        y_preds_zero_train = model_zero_train.predict(sets[i])
        y_preds_weak_train = model_weak_train.predict(sets[i])
        labels = sets[i].get_label()
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        fpr_zero, tpr_zero, thresholds_zero = roc_curve(labels, y_preds_zero_train)
        fpr_weak, tpr_weak, thresholds_weak = roc_curve(labels, y_preds_weak_train)
        results[ds]['fpr'].append(fpr)
        results[ds]['tpr'].append(tpr)
        results[ds]['thresholds'].append(thresholds)
        results[ds]['auc'].append(roc_auc_score(labels, y_preds)) 
        results_zero_train[ds]['fpr'].append(fpr_zero)
        results_zero_train[ds]['tpr'].append(tpr_zero)
        results_zero_train[ds]['thresholds'].append(thresholds_zero)
        results_zero_train[ds]['auc'].append(roc_auc_score(labels, y_preds_zero_train)) 
        results_weak_train[ds]['fpr'].append(fpr_weak)
        results_weak_train[ds]['tpr'].append(tpr_weak)
        results_weak_train[ds]['thresholds'].append(thresholds_weak)
        results_weak_train[ds]['auc'].append(roc_auc_score(labels, y_preds_weak_train))   

def convert(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(x)

with open(f"./plot/{folder_save}/results_lr_{eta}_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}.json", 'w') as outfile:
    #json.dump(results, outfile, indent = 4)
    str_j = json.dumps(results, indent = 4, sort_keys = True, default=convert)
    outfile.write(str_j)

with open(f"./plot/{folder_save}/results_zero_train_lr_{eta}_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}.json", 'w') as outfile:
    #json.dump(results, outfile, indent = 4)
    str_j = json.dumps(results_zero_train, indent = 4, sort_keys = True, default=convert)
    outfile.write(str_j)

with open(f"./plot/{folder_save}/results_weak_train_lr_{eta}_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}.json", 'w') as outfile:
    #json.dump(results, outfile, indent = 4)
    str_j = json.dumps(results_weak_train, indent = 4, sort_keys = True, default=convert)
    outfile.write(str_j)

##########################################################################################################
############## plotting the ROC curves with uncertainties ################################################
##########################################################################################################
kind = 'val'

c_fill = 'rgba(52, 152, 219, 0.2)'
c_line = 'rgba(52, 152, 219, 0.5)'
c_line_main = 'rgba(41, 128, 185, 1.0)'
c_grid = 'rgba(189, 195, 199, 0.5)'
c_annot = 'rgba(149, 165, 166, 0.5)'
c_highlight = 'rgba(192, 57, 43, 1.0)'

fpr_mean = np.linspace(0, 1, 100)

interp_tprs = []
for i in range(100):
    fpr = results[kind]['fpr'][i]
    tpr = results[kind]['tpr'][i]
    interp_tpr = np.interp(fpr_mean, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
tpr_mean = np.mean(interp_tprs, axis = 0)
tpr_mean[-1] = 1.0
tpr_std = 2*np.std(interp_tprs, axis = 0)
tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
tpr_lower = tpr_mean - tpr_std
auc = np.mean(results[kind]['auc'])

import plotly.graph_objects as go

fig = go.Figure([go.Scatter(x = tpr_upper, y = fpr_mean, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = tpr_lower, y = fpr_mean, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = tpr_mean, y = fpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.3f}')])

fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 0, y1 = 1)
fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'FPR (Background rejection)', yaxis_title = 'TPR (signal efficiency)', width = 1600, height = 900, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
fig.update_yaxes(range = [0,1], gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
fig.update_xaxes(range = [0,1], gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

fig.write_image(f"./plot/{folder_save}/ROI_simple/plotly_ROC_bg_eff_{eta}.jpg")
fig.write_image(f"./plot/{folder_save}/ROI_simple/plotly_ROC_bg_eff_{eta}.pdf")


fig = go.Figure([go.Scatter(x = 1 - fpr_mean, y = tpr_upper, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = 1 - fpr_mean, y = tpr_lower, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = 1 - fpr_mean, y = tpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.3f}')])

fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = '1 - FPR (Background rejection)', yaxis_title = 'TPR (signal efficiency)', width = 800, height = 800, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
fig.update_yaxes(range = [0,1], gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
fig.update_xaxes(range = [0,1], gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

fig.write_image(f"./plot/{folder_save}/ROI_simple/plotly_ROC_bg_rej_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.jpg")
fig.write_image(f"./plot/{folder_save}/ROI_simple/plotly_ROC_bg_rej_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.pdf")

##################################################################################################
########## Actual hyperparameter tuning ##########################################################
##################################################################################################

trials = Trials()

best_hyperparams = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)

##################################################################################################
##################################################################################################
##################################################################################################














from sklearn.metrics import accuracy_score

### Init classifier
xgb_cl = xgb.XGBClassifier(booster = 'gbtree', learning_rate = best_hyperparams['learning_rate'], gamma = best_hyperparams['gamma'], reg_alpha = best_hyperparams['reg_alpha'], reg_lambda = best_hyperparams['reg_lambda'], n_estimators = 200, max_depth = int(best_hyperparams['max_depth']), subsample = best_hyperparams['subsample'], min_child_weight = best_hyperparams['min_child_weight'], colsample_bytree = best_hyperparams['colsample_bytree'])
#xgb_cl = xgb.XGBClassifier(booster = 'gbtree', learning_rate = 0.0292, gamma = 1.087, reg_alpha = 42.0, reg_lambda = 0.381, n_estimators = 200, max_depth = 8, subsample = 0.841, min_child_weight = 2.0, colsample_bytree = 0.994)


### Fit
dtest = xgb.DMatrix(X_test, label = y_test)
#print(dtest)
dtrain = xgb.DMatrix(X_train[:int(len(X_train)*0.8), :], label = y_train[:int(len(y_train)*0.8)])
dval = xgb.DMatrix(X_train[int(len(X_train)*0.8):, :], label = y_train[int(len(y_train)*0.8):])
model_xgb = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                  verbose_eval = 1, early_stopping_rounds = 30, num_boost_round = 200) #num_boost_round = 1000,
sets = [dtrain, dval, dtest]
results_new = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
params_new = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss'}

for i, ds in enumerate(results_new.keys()):
    print(i)
    y_preds_new = model_xgb.predict(sets[i])
    labels_new = sets[i].get_label()
    fpr_new, tpr_new, thresholds_new = roc_curve(labels_new, y_preds_new)
    results_new[ds]['fpr'].append(fpr_new)
    results_new[ds]['tpr'].append(tpr_new)
    results_new[ds]['thresholds'].append(thresholds_new)
    results_new[ds]['auc'].append(roc_auc_score(labels_new, y_preds_new))  

xgb_cl.fit(X_train, y_train)

print(xgb_cl)
### Predict
preds = xgb_cl.predict(X_test)

print(accuracy_score(y_test, preds))

print(y_test)
print(model_xgb.predict(dtest))
print(np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb.predict(dtest)]))
predict_train = np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb.predict(dtest)])

print(accuracy_score(y_test, predict_train))

from xgboost import plot_importance
from xgboost import plot_tree, to_graphviz

importances = pd.DataFrame({'Feature': X.select_dtypes(include = "number").columns, 'Importance': xgb_cl.feature_importances_})
importances = importances.sort_values(by = "Importance", ascending = False)
importances = importances.set_index('Feature')
print(importances)
importances.plot.bar()

fig, ax = plt.subplots(figsize=(17,12))
plot_importance(xgb_cl, fmap = 'feature_map.txt', ax = ax)
plt.xlabel('Feature scores')
plt.ylabel("Feature names")
plt.title('Importance plot')
plt.legend([''])
#plt.show()
plt.savefig(f"./plot/{folder_save}/importance_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.jpg")

feature_importance = model.get_score(importance_type = 'weight')
keys = list(feature_importance.keys())
names_sig = ['Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading'] 
values = list(feature_importance.values())
data = pd.DataFrame(data = values, index = names_sig, columns = ['score']).sort_values(by = 'score', ascending = False)
print(data)
print(data.index)


fig = plt.figure(figsize=(17,12))
ax1 = fig.add_subplot(1,2,1)
ax1.set_axis_off()
ax2 = fig.add_subplot(1,2,2)
ax2.barh(list(reversed(data.index)), list(reversed(data.score)))
ax2.set_xlabel('Feature scores')
ax2.set_ylabel("Feature names")
ax2.set_title('Importance plot')
#plt.show()
plt.savefig(f"./plot/{folder_save}/ROI_simple/importance_train_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.jpg")

plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"./plot/{folder_save}/ROI_simple/boost_tree_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''

plt.figure(figsize=(17,12))
plot_tree(model_xgb, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"./plot/{folder_save}/ROI_simple/boost_tree_train_bg_{bgs[bg_choice]}_{bgs[bg_choice_2]}_{bgs[bg_choice_3]}_{eta}.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''
'''
plt.figure(figsize=(17,12))
to_graphviz(model_xgb, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree_train_graphviz.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''
