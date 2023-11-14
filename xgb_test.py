from coffea.util import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mplhep as hep
import hist
import argparse, sys, os, arrow, glob, yaml
from matplotlib.offsetbox import AnchoredText

from BTVNanoCommissioning.utils.plot_utils import (
    plotratio,
   
)
net_path = "/net/scratch_cms3a/vaulin/"
folder_save = 'eval_23_03_14'
if not os.path.exists(f"./plot/{folder_save}"):
    os.mkdir(f"./plot/{folder_save}")
if not os.path.exists(net_path + f"plot/{folder_save}"):
    os.mkdir(net_path + f"plot/{folder_save}")
def autoranger(array):
    val, axis = array, np.arange(0,len(array)+1)
    for i in range(len(val)):
        if val[i] != 0:
            mins = i
            break
    for i in reversed(range(len(val))):
        if val[i] != 0:
            maxs = i + 1
            break
    print(axis[mins], axis[maxs])
    return axis[mins], axis[maxs], np.max(val), np.min(val)
names_sig = ['wei','Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading'] 

#########################################################################
######### Reading the bg as output and signal as signal #################
#########################################################################
output_names = ['output_vhcc_zll_v81_bg_5_files_1_chunk', 'output_vhcc_zll_v81_bg_5_files_2_chunk', 'output_vhcc_zll_v81_bg_5_files_3_chunk', 'output_vhcc_zll_v81_bg_5_files_4_chunk', 'output_vhcc_zll_v81_bg_5_files_5_chunk', 'output_vhcc_zll_v81_bg_5_files_6_chunk', 'output_vhcc_zll_v81_bg_5_files_7_chunk', 'output_vhcc_zll_v81_bg_5_files_8_chunk', 'output_vhcc_zll_v81_bg_5_files_9_chunk', 'output_vhcc_zll_v81_bg_6_files_10_chunk']
signal_names = ['output_vhcc_zll_v54_signal_35_files']

outputs = [load(f"{name}/output.coffea") for name in output_names]
signals = [load(f"{name}/output.coffea") for name in signal_names]

outputs = [out['DYJetsToLL_nlo_vau_bg'] for out in outputs]
signals = [sig['ZHToCC_vau_sig'] for sig in signals]


output=load('output_vhcc_zll_v47_bg_2_files/output.coffea')
signal=load('output_vhcc_zll_v54_signal_5_files/output.coffea')  

output = output['DYJetsToLL_nlo_vau_bg']
signal = signal['ZHToCC_vau_sig']
#print(output['array'])
#########################################################################
########## Testing bg to see the structure ##############################
#########################################################################
for f in output['array'].keys():
    print(f)
    try:
        for k in f.keys():
            print(k)
    except AttributeError:
      print("No keys found")
#########################################################################


#########################################################################
######### Testing sig to see the structure ##############################
#########################################################################
for f in signal['array'].keys():
    print(f)
    try:
        for k in f.keys():
            print(k)
    except AttributeError:
      print("No keys found")
#########################################################################

#########################################################################
###### Reading the arrays into collect_var dictionary for sig ###########
#########################################################################
names_sig = ['wei','Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading'] 

def output_collect_sig(sig):  
    sumw_sig = {}
    collect_var_sig={}
    varlist_sig = ['weight']

    names_sig = ['wei','Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading']
    for name in names_sig:
        varlist_sig.append(f'{name}_low_ee')
        varlist_sig.append(f'{name}_high_ee')
        varlist_sig.append(f'{name}_low_mumu')
        varlist_sig.append(f'{name}_high_mumu')


    for s in sig['array'].keys():
        #iterated samples inside coffea file
        if s not in sumw_sig.keys():sumw_sig[s]=sig['array'][s]['sumw']
        else:sumw_sig[s] += sig['array'][s]['sumw']
    
        if s not in collect_var_sig.keys():collect_var_sig[s]={}
        # iterate regions(SR, CR, for H+c)
        for var in varlist_sig:
            # get arrays for each variable
            if var=='BDT' : continue
            if var not in list(collect_var_sig[s].keys()):collect_var_sig[s][var]=sig['array'][s][var].value
            else:collect_var_sig[s][var]=np.concatenate((collect_var_sig[s][var],sig['array'][s][var].value))
    
    #print(sumw)
    #print(collect_var)
    for var in collect_var_sig.keys():
        #print(var)
        for key in collect_var_sig[var].keys():
            #print(key)
            #print(collect_var_sig[var][key])
            #print(len(collect_var_sig[var][key]))
            pass
    return varlist_sig, collect_var_sig
big_signal_varlist = []
big_signal_variable_collection = []
for coffea in signals:
    varlist_sig, collect_var_sig = output_collect_sig(coffea)
    big_signal_varlist.append(varlist_sig)
    big_signal_variable_collection.append(collect_var_sig)

varlist_sig, collect_var_sig = output_collect_sig(signal)

#print(varlist_sig)
#print(big_signal_varlist)
#print(collect_var_sig)
#print(big_signal_variable_collection)
#########################################################################


#########################################################################
###### Reading the arrays into collect_var dictionary for bg ############
#########################################################################
names_bg = ['wei','Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
         'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
         'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
         'del_phi_l2_subleading', 'del_phi_l2_leading']
def output_collect_bg(bg_file): 
    sumw_bg = {}
    collect_var_bg={}
    varlist_bg = ['weight']

    names_bg = ['wei','Higgs_mass', 'Higgs_pt', 'Z_pt', 'jjVptratio', 'CvsL_max',
                'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
                'del_phi_l2_subleading', 'del_phi_l2_leading']
    for name in names_bg:
        varlist_bg.append(f'{name}_low_ee')
        varlist_bg.append(f'{name}_high_ee')
        varlist_bg.append(f'{name}_low_mumu')
        varlist_bg.append(f'{name}_high_mumu')


    for s in bg_file['array'].keys():
        #iterated samples inside coffea file
        if s not in sumw_bg.keys():sumw_bg[s]=bg_file['array'][s]['sumw']
        else:sumw_bg[s] += bg_file['array'][s]['sumw']
    
        if s not in collect_var_bg.keys():collect_var_bg[s]={}
        # iterate regions(SR, CR, for H+c)
        for var in varlist_bg:
            # get arrays for each variable
            if var=='BDT' : continue
            if var not in list(collect_var_bg[s].keys()):collect_var_bg[s][var]=bg_file['array'][s][var].value
            else:collect_var_bg[s][var]=np.concatenate((collect_var_bg[s][var],bg_file['array'][s][var].value))
    
    #print(sumw)
    #print(collect_var)
    for var in collect_var_bg.keys():
        #print(collect_var[var])
        for key in collect_var_bg[var].keys():
            print(key)
            #print(collect_var[var][key])
            #print(len(collect_var[var][key]))
    return varlist_bg, collect_var_bg
varlist_bg, collect_var_bg = output_collect_bg(output)
big_bg_varlist = []
big_bg_variable_collection = []
for coffea in outputs:
    varlist_bg, collect_var_bg = output_collect_bg(coffea)
    big_bg_varlist.append(varlist_bg)
    big_bg_variable_collection.append(collect_var_bg)

#print(varlist_bg)
#print(big_bg_varlist)
#print(collect_var_bg)
#print(big_bg_variable_collection)
#########################################################################


#########################################################################
## Mergemap - dictionary with files, associated with their categories ###
#########################################################################
mergemap={'signal': ['ZHToCC_vau_sig'], 'bg': ['DYJetsToLL_nlo_vau_bg']}
trainvar = []
#for s in varlist_sig:
#    trainvar.append(f'{s}_signal')
#for b in varlist_bg:
#    trainvar.append(f'{b}_background')
trainvar = varlist_sig
#print(trainvar)
MCvar={}
weivar={}

for var in trainvar :
    MCbkgLM = []
    MCvar[var]={}
    for m in mergemap:
        tmpml = []
        tmpwei = []
        if m == 'signal':
            for colvarsig in big_signal_variable_collection:
                tmpml=np.concatenate((tmpml,colvarsig[mergemap[m][0]][var])) 
            
                tmpwei=np.concatenate((tmpwei,colvarsig[mergemap[m][0]]['weight']))
        elif m == 'bg':
            for colvarbag in big_bg_variable_collection:
                tmpml=np.concatenate((tmpml,colvarbag[mergemap[m][0]][var])) 
            
                tmpwei=np.concatenate((tmpwei,colvarbag[mergemap[m][0]]['weight'])) 
        MCvar[var][m]=tmpml
        weivar[m]=tmpwei
        MCbkgLM+=[tmpml]

print(MCvar.keys())
print(MCvar['Higgs_mass_low_ee'].keys())
len_var = []
len_var_bg = []

df_sig = pd.DataFrame([], columns = [f'{col}_low_ee' for col in names_sig])
print(df_sig)
for var in MCvar.keys():
    if '_low_ee' in var:
        len_var.append(len(MCvar[var]['signal']))
        df_sig[var] = MCvar[var]['signal']
        df_sig['target'] = np.ones(np.max(len_var))
print(df_sig)
print(np.max(len_var), np.min(len_var))


df_bg = pd.DataFrame([], columns = [f'{col}_low_ee' for col in names_sig])
print(df_bg)
for var in MCvar.keys():
    if '_low_ee' in var:
        len_var_bg.append(len(MCvar[var]['bg']))
        df_bg[var] = MCvar[var]['bg']
        df_bg['target'] = np.zeros(np.max(len_var_bg))
print(df_bg)
print(np.max(len_var_bg), np.min(len_var_bg))

df = pd.concat([df_sig, df_bg], ignore_index = True)
print(df)
print(df.info())
df.to_csv('xgb_training_dataset_low_ee.csv', sep=',', encoding='utf-8', index=False)

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)
names_sig_updated = ['m(H)', '$p_t$(H)', '$p_t$(Z)', '$\\frac{p_t(V)}{p_t(H)}$', '$CvsL_{max}$',
                 '$CvsL_{min}$', '$CvsB_{max}$', '$CvsB_{min}$', '$p_t$ of $CvsL_{max}$ jet', '$p_t$ of $CvsL_{min}$ jet',
                 '$\Delta\Phi(V, H)$', '$\Delta R(jet_1, jet_2)$', '$\Delta\eta(jet_1, jet_2)$', '$\Delta\Phi(e_1, e_2)$', '$\Delta\eta(e_1, e_2)$',
                 '$\Delta\Phi (e_{subleading}, jet_{subleading})$', '$\Delta\Phi (e_{subleading}, jet_{leading})$'] 

c = 0
for col in names_sig[1:]:
    
    plt.figure(figsize=(10,10))
    len_sig = 0
    for i in range(0,len(df['target'])):
        if df['target'][i] == 1:
             len_sig += 1
    print(len_sig)
    names_big_ax = ['Higgs_mass', 'Higgs_pt', 'Z_pt', 'pt_lead', 'pt_sublead']
    if col in names_big_ax:
        hist.Hist.new.Regular(150, 0, 180).Double().fill(np.array(df[f'{col}_low_ee'][:len_sig])).plot()
        hist.Hist.new.Regular(150, 0, 180).Double().fill(np.array(df[f'{col}_low_ee'][len_sig:])).plot()
    else:
        hist.Hist.new.Regular(150, 0, 5).Double().fill(np.array(df[f'{col}_low_ee'][:len_sig])).plot()
        hist.Hist.new.Regular(150, 0, 5).Double().fill(np.array(df[f'{col}_low_ee'][len_sig:])).plot()
    if 'pt' in col:
        if 'ratio' not in col:
            plt.xlabel('$p_t$ in Gev')
        else:
            plt.xlabel('')
    elif 'mass' in col:
        plt.xlabel('Mass in Gev')
    else:
        plt.xlabel('')
    plt.ylabel("Counts")
    plt.title(f'{names_sig_updated[c]}_low_ee')
    plt.legend(['Signal', 'Background'])
    #plt.show()
    plt.savefig(f"plot/{folder_save}/{col}_low_ee.jpg")



    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_low_ee'][:len_sig]),bins = 80, weights = np.array(df['wei_low_ee'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        density = True,
    )
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_low_ee'][len_sig:]),bins =80, weights = np.array(df['wei_low_ee'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_low_ee'][:len_sig]),bins = 80, weights = np.array(df['wei_low_ee'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_low_ee'][len_sig:]),bins =80, weights = np.array(df['wei_low_ee'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    plt.plot(bins1[:-1], ratio, 'ko')
    plt.plot(bins1[:-1], [1]*len(ratio), '--', color = 'black')
    
    
    ##  plot settings, adjust range
    rax.set_xlabel(f'{names_sig_updated[c]} low ee')
    ax.set_xlabel(None)
    ax.set_ylabel("Events (normalised)")
    rax.set_ylabel('$\\frac{Signal}{Background}$')
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax, maxval, minval = autoranger(np.array(df[f'{col}_low_ee'][:len_sig]))
    rax.set_xlim(minval, maxval)
    at = AnchoredText(
        "",
        loc=2,
        frameon=False,
    )
    ax.add_artist(at)
    hep.mpl_magic(ax=ax)
    ax.set_ylim(bottom=0)

    logext = ""
    '''
    # log y axis
    if "log" in config.keys() and config["log"]:
        ax.set_yscale("log")
        logext = "_log"
        ax.set_ylim(bottom=0.1)
        hep.mpl_magic(ax=ax)
    if "norm" in config.keys() and config["norm"]:
        logext = "_norm" + logext
    '''
    fig.savefig(f"plot/{folder_save}/compare_{col}_low_ee.pdf")
    fig.savefig(f"plot/{folder_save}/compare_{col}_low_ee.jpg")
    c += 1
    

X = df.drop("target", axis = 1)
print(X)
X = X.drop("wei_low_ee", axis = 1)
print(X)
print(X.info())

y = df["target"]
print(y)




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

import xgboost as xgb

X_processed = full_processor.fit_transform(X)
y_processed = SimpleImputer(strategy = "most_frequent").fit_transform(y.values.reshape(-1,1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, stratify = y_processed, random_state = 1121218)

from sklearn.metrics import accuracy_score

### Init classifier
xgb_cl = xgb.XGBClassifier(booster = 'gbtree', base_score = 0.5, learning_rate = 0.01, gamma = 1, reg_alpha = 0.2, reg_lambda = 0.2, n_estimators = 1000, max_depth = 3, subsample = 0.8)

### Fit
xgb_cl.fit(X_train, y_train)

print(xgb_cl)
### Predict
preds = xgb_cl.predict(X_test)

print(accuracy_score(y_test, preds))

from xgboost import plot_importance
from xgboost import plot_tree

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

plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"plot/{folder_save}/boost_tree.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()
