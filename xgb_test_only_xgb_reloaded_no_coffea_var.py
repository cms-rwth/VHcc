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

net_path = "/net/scratch_cms3a/vaulin/"
folder_save = 'eval_23_06_26_2'
if not os.path.exists(f"./plot/{folder_save}"):
    os.mkdir(f"./plot/{folder_save}")
if not os.path.exists(net_path + f"plot/{folder_save}"):
    os.mkdir(net_path + f"plot/{folder_save}")
df = pd.read_csv(net_path + 'xgb_training_dataset_low_mumu.csv')

roi = 'low_mumu'
learning_rate = 0.3

names_sig = ['Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading']

var = f'del_phi_l2_subleading_{roi}'

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)


df  = df.sample(frac = 1).reset_index(drop=True)

X = df[var]
print(X)
print(X.info())

X_signal = df[var][df.target == 1]
X_bg = df[var][df.target == 0]

y = df["target"]
print(y)

y_signal = df["target"][df.target == 1]
y_bg = df["target"][df.target == 0]

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy = "most_frequent")), ("oh-encode", OneHotEncoder(handle_unknown = "ignore", sparse = False)),])

from sklearn.preprocessing import StandardScaler
numeric_pipeline = Pipeline(steps = [("impute", SimpleImputer(strategy = "mean")), ("scale", StandardScaler())])

cat_cols = pd.Index([], dtype = 'object')
num_cols = pd.Index([var], dtype = 'object')

print(cat_cols)
print(num_cols)

from sklearn.compose import ColumnTransformer

full_processor = ColumnTransformer(transformers = [("numeric", numeric_pipeline, num_cols), ("categorical", categorical_pipeline, cat_cols),])



#X_processed = full_processor.fit_transform(X)
y_processed = SimpleImputer(strategy = "most_frequent").fit_transform(y.values.reshape(-1,1))
y_processed_sig = SimpleImputer(strategy = "most_frequent").fit_transform(y_signal.values.reshape(-1,1))
y_processed_bg = SimpleImputer(strategy = "most_frequent").fit_transform(y_bg.values.reshape(-1,1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_processed, stratify = y_processed, random_state = 1121218)
X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X_signal, y_processed_sig, stratify = y_processed_sig, random_state = 1121218)
X_train_bg, X_test_bg, y_train_bg, y_test_bg = train_test_split(X_bg, y_processed_bg, stratify = y_processed_bg, random_state = 1121218)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101, stratify = y)
print(X_train)
print(X_test)
print(y_train)


###############################################################################################################################################
################### Getting ROC curves from json files ########################################################################################
###############################################################################################################################################
def convert(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(x)


kind = 'val'
#kind = 'test'
#kind = 'train'

def pretty_ROC_Curve(tr_set, kind, type, var):

    with open(tr_set) as user_file:
        file_contents = user_file.read() 
 
    results = json.loads(file_contents)
    params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': learning_rate}
    metrics = ['auc', 'fpr', 'tpr', 'thresholds']
    
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


    fig = go.Figure([go.Scatter(x = tpr_upper, y = 1 - fpr_mean, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = tpr_lower, y = 1 - fpr_mean, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = tpr_mean, y = 1 - fpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.5f}')])

    fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
    fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'TPR (signal efficiency)', yaxis_title = '1 - FPR (Background rejection)', width = 800, height = 800, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
    fig.update_yaxes(range = range_plot_y, gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
    fig.update_xaxes(range = range_plot_x, gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

    fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}_{var}.jpg")
    fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}_{var}.pdf")


pretty_ROC_Curve(f"plot/{folder_save}/results_lr_{learning_rate}.json", kind, "full", var)
##############################################################################################################################################################
##################### Zero train ROC #########################################################################################################################
##############################################################################################################################################################

pretty_ROC_Curve(f"plot/{folder_save}/results_zero_train_lr_{learning_rate}.json", kind, 'zero', var)

##############################################################################################################################################################
##################### Weak train ROC #########################################################################################################################
##############################################################################################################################################################

pretty_ROC_Curve(f"plot/{folder_save}/results_weak_train_lr_{learning_rate}.json", kind, 'weak', var)

##############################################################################################################################################################


trials = Trials()

##############################################################################################################################################################
##################### Initiate the final training to be presented with the best parameters ###################################################################
##############################################################################################################################################################

from sklearn.metrics import accuracy_score

### Init classifier
xgb_cl = xgb.XGBClassifier(booster = 'gbtree', learning_rate = 0.0292, gamma = 1.087, reg_alpha = 42.0, reg_lambda = 0.381, n_estimators = 200, max_depth = 8, subsample = 0.841, min_child_weight = 2.0, colsample_bytree = 0.994, scale_pos_weight = 10)

### Fit
params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': learning_rate}
metrics = ['auc', 'fpr', 'tpr', 'thresholds']
dtest = xgb.DMatrix(X_test, label = y_test)
dtest_signal = xgb.DMatrix(X_test_sig, label = y_test_sig)
dtest_bg = xgb.DMatrix(X_test_bg, label = y_test_bg)
#print(dtest)
dtrain = xgb.DMatrix(X_train[:int(len(X_train)*0.8)], label = y_train[:int(len(y_train)*0.8)])
dval = xgb.DMatrix(X_train[int(len(X_train)*0.8):], label = y_train[int(len(y_train)*0.8):])
model_xgb = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                  verbose_eval = 1, early_stopping_rounds = 30, num_boost_round = 200) #num_boost_round = 1000,
model_xgb_weak = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                  verbose_eval = 1, early_stopping_rounds = 30, num_boost_round = 20) #num_boost_round = 1000,
model_xgb_zero = xgb.train(dtrain = dtrain, params = params, evals = [(dtrain, 'train'),(dval, 'dval')],
                  verbose_eval = 1, early_stopping_rounds = 30, num_boost_round = 2) #num_boost_round = 1000,
sets = [dtrain, dval, dtest]
results_new = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
results_new_weak = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
results_new_zero = {'train': {m:[] for m in metrics},
           'val': {m:[] for m in metrics},
           'test': {m:[] for m in metrics}}
params_new = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss'}

for i, ds in enumerate(results_new.keys()):
    print(i)
    y_preds_new = model_xgb.predict(sets[i])
    y_preds_new_weak = model_xgb_weak.predict(sets[i])
    y_preds_new_zero = model_xgb_zero.predict(sets[i])
    labels_new = sets[i].get_label()
    fpr_new, tpr_new, thresholds_new = roc_curve(labels_new, y_preds_new)
    fpr_new_weak, tpr_new_weak, thresholds_new_weak = roc_curve(labels_new, y_preds_new_weak)
    fpr_new_zero, tpr_new_zero, thresholds_new_zero = roc_curve(labels_new, y_preds_new_zero)
    results_new[ds]['fpr'].append(fpr_new)
    results_new[ds]['tpr'].append(tpr_new)
    results_new[ds]['thresholds'].append(thresholds_new)
    results_new[ds]['auc'].append(roc_auc_score(labels_new, y_preds_new))
    results_new_weak[ds]['fpr'].append(fpr_new_weak)
    results_new_weak[ds]['tpr'].append(tpr_new_weak)
    results_new_weak[ds]['thresholds'].append(thresholds_new_weak)
    results_new_weak[ds]['auc'].append(roc_auc_score(labels_new, y_preds_new_weak)) 
    results_new_zero[ds]['fpr'].append(fpr_new_zero)
    results_new_zero[ds]['tpr'].append(tpr_new_zero)
    results_new_zero[ds]['thresholds'].append(thresholds_new_zero)
    results_new_zero[ds]['auc'].append(roc_auc_score(labels_new, y_preds_new_zero))  

def pretty_ROC_Curve_var(results, kind, type, var):
 
    results = results
    params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': learning_rate}
    metrics = ['auc', 'fpr', 'tpr', 'thresholds']
    
    c_fill = 'rgba(52, 152, 219, 0.2)'
    c_line = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid = 'rgba(189, 195, 199, 0.5)'
    c_annot = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'

    fpr_mean = np.linspace(0, 1, 100)

    interp_tprs = []
    for i in range(1):
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


    fig = go.Figure([go.Scatter(x = tpr_upper, y = 1 - fpr_mean, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'upper'),
                 go.Scatter(x = tpr_lower, y = 1 - fpr_mean, fill = 'tonexty', fillcolor = c_fill, line = dict(color = c_line, width = 1), hoverinfo = 'skip', showlegend = False, name = 'lower'),
                 go.Scatter(x = tpr_mean, y = 1 - fpr_mean, line = dict(color = c_line_main, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.5f}')])

    fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
    fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'TPR (signal efficiency)', yaxis_title = '1 - FPR (Background rejection)', width = 800, height = 800, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
    fig.update_yaxes(range = range_plot_y, gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
    fig.update_xaxes(range = range_plot_x, gridcolor = c_grid, constrain = 'domain', linecolor = 'black') 

    fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}_{var}_new.jpg")
    fig.write_image(f"plot/{folder_save}/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_kind_{kind}_{var}_new.pdf") 

def pretty_ROC_Curve_var_test_train_val(results, type, var):
 
    results = results
    params = {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': learning_rate}
    metrics = ['auc', 'fpr', 'tpr', 'thresholds']
    
    c_fill = 'rgba(52, 152, 219, 0.2)'
    c_line = 'rgba(52, 152, 219, 0.5)'
    c_line_train = 'rgba(41, 128, 185, 1.0)'
    c_line_test = 'rgba(58, 217, 19, 0.8)'
    c_line_val = 'rgba(244, 70, 10, 0.8)'
    c_grid = 'rgba(189, 195, 199, 0.5)'
    c_annot = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'

    fpr_mean = np.linspace(0, 1, 100)

    interp_tprs = []

    range_plot_x = [0,1]
    range_plot_y = [0.2,1]

    import plotly.graph_objects as go
    colours = {'test':c_line_test, 'train': c_line_train, 'val': c_line_val}
    fig_test = 0
    fig_train = 0
    fig_val = 0
    figs = {'test': fig_test, 'train': fig_train, 'val': fig_val}
    for kind in ['test', 'val', 'train']:
        for i in range(1):
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
        colour = colours[kind]
    

        figs[kind] = go.Scatter(x = tpr_mean, y = 1 - fpr_mean, line = dict(color = colour, width = 2), hoverinfo = 'skip', showlegend = True, name = f'AUC: {auc:.5f}, {kind}')
    fig = go.Figure(data = [figs['test'], figs['train'], figs['val']])

    fig.add_shape(type = 'line', line = dict(dash = 'dash'), x0 = 0, x1 = 1, y0 = 1, y1 = 0)
    fig.update_layout(template = 'plotly_white', title_x = 0.5, xaxis_title = 'TPR (signal efficiency)', yaxis_title = '1 - FPR (Background rejection)', width = 800, height = 800, legend = dict( yanchor = 'bottom', xanchor = 'right', x = 0.95, y = 0.01,))
    fig.update_yaxes(range = range_plot_y, gridcolor = c_grid, scaleanchor = 'x', scaleratio = 1, linecolor = 'black')
    fig.update_xaxes(range = range_plot_x, gridcolor = c_grid, constrain = 'domain', linecolor = 'black')
    
    if not os.path.exists(f"plot/{folder_save}/ROC"):
        os.mkdir(f"plot/{folder_save}/ROC") 

    fig.write_image(f"plot/{folder_save}/ROC/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_all_{var}_new.jpg")
    fig.write_image(f"plot/{folder_save}/ROC/plotly_ROC_bg_rej_reloaded_{type}_lr_{learning_rate}_rangex_{range_plot_x}_rangey_{range_plot_y}_all_{var}_new.pdf") 

pretty_ROC_Curve_var(results_new, 'test', 'full', var)

pretty_ROC_Curve_var_test_train_val(results_new, 'full', var)

xgb_cl.fit(X_train, y_train)

print(xgb_cl)

###################################################################################################################################
################################## Predict and give the final accuracy scores and importance plots ################################
###################################################################################################################################
preds = xgb_cl.predict(X_test)

print(accuracy_score(y_test, preds))

print(y_test)
print(model_xgb.predict(dtest))
print(np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb.predict(dtest)]))
predict_train = np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb.predict(dtest)])
predict_train_weak = np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb_weak.predict(dtest)])
predict_train_zero = np.array([1 if dtest_val > 0.5 else 0 for dtest_val in model_xgb_zero.predict(dtest)])

print(accuracy_score(y_test, predict_train))

from xgboost import plot_importance
from xgboost import plot_tree, to_graphviz

#importances = pd.DataFrame({'Feature': X.select_dtypes(include = "number").columns, 'Importance': xgb_cl.feature_importances_})
#importances = importances.sort_values(by = "Importance", ascending = False)
#importances = importances.set_index('Feature')
#print(importances)
#importances.plot.bar()

fig, ax = plt.subplots(figsize=(17,12))
plot_importance(xgb_cl, fmap = 'feature_map_var.txt', ax = ax)
plt.xlabel('Feature scores')
plt.ylabel("Feature names")
plt.title('Importance plot')
plt.legend([''])
#plt.show()
plt.savefig(f"plot/{folder_save}/importance_{var}.jpg")


feature_importance = model_xgb.get_score(importance_type = 'weight')
keys = list(feature_importance.keys())
'''names_sig = ['m(H)', '$p_t$(H)', '$p_t$(Z)', '$\\frac{p_t(V)}{p_t(H)}$', '$CvsL_{max}$',
                 '$CvsL_{min}$', '$CvsB_{max}$', '$CvsB_{min}$', '$p_t$ of $CvsL_{max}$ jet', '$p_t$ of $CvsL_{min}$ jet',
                 '$\Delta\Phi(V, H)$', '$\Delta R(jet_1, jet_2)$', '$\Delta\eta(jet_1, jet_2)$', '$\Delta\Phi(jet_1, jet_2)$', '$\Delta\Phi(l_1, l_2)$', '$\Delta\eta(l_1, l_2)$',
                 '$\Delta\Phi (l_{subleading}, jet_{subleading})$', '$\Delta\Phi (l_{subleading}, jet_{leading})$']'''
names_sig = ['$\Delta\Phi (l_{subleading}, jet_{subleading})$'] 
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
plt.savefig(f"plot/{folder_save}/importance_train_lr_{learning_rate}_{var}.jpg")


plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map_var.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree_{var}.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''

plt.figure(figsize=(17,12))
plot_tree(model_xgb, fmap = 'feature_map_var.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree_train_lr_{learning_rate}_{var}.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''

plt.figure(figsize=(17,12))
plt.hist(np.array(model_xgb.predict(dtest)), bins = 40, edgecolor = 'blue',fill = False)
plt.hist(np.array(predict_train), bins = 40, edgecolor = 'green', hatch = '/', fill = False)
plt.hist(y_test, bins = 40, facecolor = 'orange', edgecolor = 'orange', fill = False)
plt.title('Classifier output')
plt.legend(['Train output', 'Train output after threshold','Test data'])
#plt.show()
plt.savefig(f"plot/{folder_save}/class_output_train_lr_{learning_rate}_{var}.jpg")

plt.figure(figsize=(17,12))
plt.hist(np.array(model_xgb.predict(dtest_signal)), bins = 40, edgecolor = 'blue',fill = False)
plt.hist(np.array(model_xgb.predict(dtest_bg)), bins = 40, edgecolor = 'red', fill = False)
plt.title('Classifier output')
plt.legend(['Signal', 'Background'])
#plt.show()
plt.savefig(f"plot/{folder_save}/class_output_train_lr_{learning_rate}_{var}_sig_vs_bg.jpg")

plt.figure(figsize=(17,12))
plt.hist(np.array(model_xgb_weak.predict(dtest)), bins = 40, edgecolor = 'blue',fill = False)
plt.hist(np.array(predict_train_weak), bins = 40, edgecolor = 'green', hatch = '/', fill = False)
plt.hist(y_test, bins = 40, facecolor = 'orange', edgecolor = 'orange', fill = False)
plt.title('Classifier output')
plt.legend(['Train output', 'Train output after threshold','Test data'])
#plt.show()
plt.savefig(f"plot/{folder_save}/class_output_train_lr_{learning_rate}_{var}_weak.jpg")

plt.figure(figsize=(17,12))
plt.hist(np.array(model_xgb_zero.predict(dtest)), bins = 40, edgecolor = 'blue',fill = False)
plt.hist(np.array(predict_train_zero), bins = 40, edgecolor = 'green', hatch = '/', fill = False)
plt.hist(y_test, bins = 40, facecolor = 'orange', edgecolor = 'orange', fill = False)
plt.title('Classifier output')
plt.legend(['Train output', 'Train output after threshold','Test data'])
#plt.show()
plt.savefig(f"plot/{folder_save}/class_output_train_lr_{learning_rate}_{var}_zero.jpg")

###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''

with open(f"plot/{folder_save}/ROC.txt", "a") as myfile:
            myfile.write(f"ROC score for {var}: " + str(accuracy_score(y_test, predict_train)) + "  " + '\n')

'''
plt.figure(figsize=(17,12))
to_graphviz(model_xgb, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree_train_graphviz.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()'''
