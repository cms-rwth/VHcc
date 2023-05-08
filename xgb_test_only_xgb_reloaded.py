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

folder_save = 'eval_23_03_07_1'
if not os.path.exists(f"./plot/{folder_save}"):
    os.mkdir(f"./plot/{folder_save}")
df = pd.read_csv('xgb_training_dataset_low_ee.csv')


learning_rate = 0.3

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)

X = df.drop("target", axis = 1)
print(X)
X = X.drop("wei_low_ee", axis = 1)
print(X)
print(X.info())

y = df["target"]
print(y)

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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, stratify = y_processed, random_state = 1121218)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101, stratify = y)
print(X_train)
print(X_test)
print(y_train)

with open(f"plot/{folder_save}/results_lr_{learning_rate}.json") as user_file:
    file_contents = user_file.read()

results = json.loads(file_contents)
params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': learning_rate}
metrics = ['auc', 'fpr', 'tpr', 'thresholds']
def convert(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(x)


#kind = 'val'
kind = 'test'
#kind = 'train'

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

range_plot_x = [0,1]
range_plot_y = [0.2,1]

import plotly.graph_objects as go

fig = go.Figure([go.Scatter(x = tpr_upper, y = fpr_mean, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = tpr_lower, y = fpr_mean, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = tpr_mean, y = fpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.3f}')])

fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'TPR (signal efficiency)', yaxis_title = 'FPR (Background efficiency)', width = 1600, height = 900, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
fig.update_yaxes(range = range_plot_y, gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
fig.update_xaxes(range = range_plot_x, gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_eff_reloaded__lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}.jpg")
fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_eff_reloaded__lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}.pdf")


fig = go.Figure([go.Scatter(x = tpr_upper, y = 1 - fpr_mean, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = tpr_lower, y = 1 - fpr_mean, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = tpr_mean, y = 1 - fpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.3f}')])

fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'TPR (signal efficiency)', yaxis_title = '1 - FPR (Background rejection)', width = 800, height = 800, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
fig.update_yaxes(range = range_plot_y, gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
fig.update_xaxes(range = range_plot_x, gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded__lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}.jpg")
fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded__lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}.pdf")




trials = Trials()

#best_hyperparams = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 100, trials = trials)
#print("The best hyperparameters are: ", "\n")
#print(best_hyperparams)
















from sklearn.metrics import accuracy_score

### Init classifier
#xgb_cl = xgb.XGBClassifier(booster = 'gbtree', learning_rate = best_hyperparams['learning_rate'], gamma = best_hyperparams['gamma'], reg_alpha = best_hyperparams['reg_alpha'], reg_lambda = best_hyperparams['reg_lambda'], n_estimators = 200, max_depth = int(best_hyperparams['max_depth']), subsample = best_hyperparams['subsample'], min_child_weight = best_hyperparams['min_child_weight'], colsample_bytree = best_hyperparams['colsample_bytree'])
xgb_cl = xgb.XGBClassifier(booster = 'gbtree', learning_rate = 0.0292, gamma = 1.087, reg_alpha = 42.0, reg_lambda = 0.381, n_estimators = 200, max_depth = 8, subsample = 0.841, min_child_weight = 2.0, colsample_bytree = 0.994)

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
plt.savefig(f"plot/{folder_save}/importance.jpg")


feature_importance = model_xgb.get_score(importance_type = 'weight')
keys = list(feature_importance.keys())
names_sig = ['m(H)', '$p_t$(H)', '$p_t$(Z)', '$\\frac{p_t(V)}{p_t(H)}$', '$CvsL_{max}$',
                 '$CvsL_{min}$', '$CvsB_{max}$', '$CvsB_{min}$', '$p_t$ of $CvsL_{max}$ jet', '$p_t$ of $CvsL_{min}$ jet',
                 '$\Delta\Phi(V, H)$', '$\Delta R(jet_1, jet_2)$', '$\Delta\eta(jet_1, jet_2)$', '$\Delta\Phi(e_1, e_2)$', '$\Delta\eta(e_1, e_2)$',
                 '$\Delta\Phi (e_{subleading}, jet_{subleading})$', '$\Delta\Phi (e_{subleading}, jet_{leading})$'] 
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
plt.savefig(f"plot/{folder_save}/importance_train_lr_{learning_rate}.jpg")


plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''

plt.figure(figsize=(17,12))
plot_tree(model_xgb, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree_train_lr_{learning_rate}.jpg", dpi = 1800)
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
