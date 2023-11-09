from coffea.util import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mplhep as hep
import hist
import argparse, sys, os, arrow, glob, yaml
from matplotlib.offsetbox import AnchoredText
from pathlib import Path
import os
from BTVNanoCommissioning.utils.plot_utils import (
    plotratio,
   
)
net_path = "/net/scratch_cms3a/vaulin/"
folder_save = 'eval_23_08_08'
if not os.path.exists(f"./plot/{folder_save}"):
    os.mkdir(f"./plot/{folder_save}")
if not os.path.exists(f"./plot/{folder_save}/Small_scale"):
    os.mkdir(f"./plot/{folder_save}/Small_scale")
if not os.path.exists(f"./plot/{folder_save}/Big_scale"):
    os.mkdir(f"./plot/{folder_save}/Big_scale")
if not os.path.exists(f"./plot/{folder_save}/Small_but_not_that_small_scale"):
    os.mkdir(f"./plot/{folder_save}/Small_but_not_that_small_scale")
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
names_sig = ['wei', 'Higgs_mass', 'Higgs_pt', 'Z_pt', 'Z_mass', 'Z_pt_gen', 'Z_mass_gen', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading']

names_sig_data = ['wei', 'Higgs_mass', 'Higgs_pt', 'Z_pt', 'Z_mass', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading'] 

roiis = ['high_mumu', 'high_ee', 'low_mumu', 'low_ee']
roi = 'low_mumu'
######################################################################################
##### Read np arrays of signal sample ################################################
######################################################################################
data_path = 'condor_signal_06_mid/'
paths_np = [str(x) for x in Path(data_path + "ZHToCC_vau_sig").glob("**/*.npy") if ("_full" in str(x))] 
#print(paths_np)
print(len(paths_np))
df_sig_full_np = pd.DataFrame([], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
print(df_sig_full_np)

key_np = {}
for col in names_sig:
    for rois in roiis:
        key_np[f'{col}_{rois}'] = []
for col in names_sig:
    for rois in roiis:
        for path in paths_np:
            if f'{col}_{rois}' in path:
                key_np[f'{col}_{rois}'].append(path)

for key in key_np.keys():
    #print(len(key_np[key]) == len(set(key_np[key])))
    key_np[key] = [np.load(element) for element in key_np[key]]
    #print(key)
    
print(key_np)

key_np_full = {}
max_length = 0
for col in names_sig:
    for rois in roiis:
        key_np_full[f'{col}_{rois}'] = np.array([])
print(key_np_full)
for key in key_np_full.keys():
    key_np_full[key] = np.concatenate(tuple(key_np[key]), axis = None)
    print(len(key_np_full[key]))
    if max_length < len(key_np_full[key]):
        max_length = len(key_np_full[key])

for key in key_np_full.keys():                 
    #df_sig_full_np[key] = pd.Series(key_np_full[key])
    df_sig_full_np[key] = list(np.append(key_np_full[key], np.repeat(np.nan, max_length- (len(key_np_full[key])))))
#df_sig_full_np = pd.DataFrame([pd.Series(key_np_full[key]) for key in key_np_full.keys()], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
print(df_sig_full_np)
df_s_new_np = df_sig_full_np[[f'{col}_{roi}' for col in names_sig]]

print(len(df_s_new_np[f"wei_{roi}"]))
our_aray_results = len(df_s_new_np[f"wei_{roi}"])



df_s_new_np = df_s_new_np.dropna()
print(df_s_new_np)
len_var = []
for col in names_sig:
    len_var.append(len(df_s_new_np[f'{col}_{roi}']))
    df_s_new_np['target'] = np.ones(np.max(len_var))
print(df_s_new_np)


df_s_new_np.to_csv(f'./plot/{folder_save}/numpy_data_signal.csv', sep=',', encoding='utf-8', index=False)
#df_s_new_np = pd.read_csv(f'./plot/{folder_save}/numpy_data.csv', sep=',', encoding='utf-8')
######################################################################################


######################################################################################
##### Read np arrays of background sample ############################################
######################################################################################
data_path = 'condor_back_07_early/'
#paths_np_back = [str(x) for x in Path(data_path + "DYJetsToLL_nlo_vau_bg").glob("**/*.npy") if ("_full" in str(x))] 
paths_np_back = [str(x) for x in Path(data_path + "TTTo2L2Nu_vau_bg").glob("**/*.npy") if ("_full" in str(x))] 
#paths_np_back = [str(x) for x in Path("./condor_back_04_mid/DYJetsToLL_nlo_vau_bg").glob("**/*.npy") if ("_full" in str(x))] 
#print(paths_np_back)TTTo2L2Nu_vau_bg
print(len(paths_np_back))
df_back_full_np = pd.DataFrame([], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
print(df_back_full_np)

key_np_back = {}
for col in names_sig:
    for rois in roiis:
        key_np_back[f'{col}_{rois}'] = []
for col in names_sig:
    for rois in roiis:
        for path in paths_np_back:
            if f'{col}_{rois}' in path:
                key_np_back[f'{col}_{rois}'].append(path)
#print(key_np_back)
for key in key_np_back.keys():
    print(len(key_np_back[key]) == len(set(key_np_back[key])))
    key_np_back[key] = [np.load(element) for element in key_np_back[key]]
    print(key)

#print(key_np_back)

max_length_back = 0
key_np_full_back = {}
for col in names_sig:
    for rois in roiis:
        key_np_full_back[f'{col}_{rois}'] = np.array([])
for key in key_np_full_back.keys():
    key_np_full_back[key] = np.concatenate(tuple(key_np_back[key]), axis = None)
    print(len(key_np_full_back[key]))
    if max_length_back < len(key_np_full_back[key]):
        max_length_back = len(key_np_full_back[key])
#print(key_np_full_back)

for key in key_np_full_back.keys():                 
    #df_sig_full_np[key] = pd.Series(key_np_full[key])
    df_back_full_np[key] = list(np.append(key_np_full_back[key], np.repeat(np.nan, max_length_back- (len(key_np_full_back[key])))))
#df_sig_full_np = pd.DataFrame([pd.Series(key_np_full[key]) for key in key_np_full.keys()], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
print(df_back_full_np)
df_b_full_np = df_back_full_np[[f'{col}_{roi}' for col in names_sig]]
df_b_new_np = df_b_full_np.dropna()
print(df_b_new_np)

len_var = []
for col in names_sig:
    len_var.append(len(df_b_new_np[f'{col}_{roi}']))
    df_b_new_np['target'] = np.zeros(np.max(len_var))
print(df_b_new_np)
df_b_new_np.to_csv(f'./plot/{folder_save}/numpy_data_bg.csv', sep=',', encoding='utf-8', index=False)
######################################################################################

######################################################################################
##### Read np arrays of data sample ##################################################
######################################################################################
data_path = 'condor_back_08_early/'
datas = ["Run2017B_DoubleMu_vau", "Run2017D_DoubleMu_vau", "Run2017E_DoubleMu_vau", "Run2017F_DoubleMu_vau"] #"Run2017C_DoubleMu_vau"
df_data = pd.DataFrame([], columns = [f'{col}_{rois}' for col in names_sig_data for rois in roiis])
for data in datas:
    paths_np_data = [str(x) for x in Path(data_path + data).glob("**/*.npy") if ("_full" in str(x))] 

    print(len(paths_np_data))
    df_data_full_np = pd.DataFrame([], columns = [f'{col}_{rois}' for col in names_sig_data for rois in roiis])
    print(df_data_full_np)

    key_np_data = {}
    for col in names_sig_data:
        for rois in roiis:
            key_np_data[f'{col}_{rois}'] = []
    for col in names_sig_data:
        for rois in roiis:
            for path in paths_np_data:
                if f'{col}_{rois}' in path:
                    key_np_data[f'{col}_{rois}'].append(path)
    #print(key_np_back)
    for key in key_np_data.keys():
        print(len(key_np_data[key]) == len(set(key_np_data[key])))
        key_np_data[key] = [np.load(element) for element in key_np_data[key]]
        print(key)

    #print(key_np_back)

    max_length_data = 0
    key_np_full_data = {}
    for col in names_sig_data:
        for rois in roiis:
            key_np_full_data[f'{col}_{rois}'] = np.array([])
    for key in key_np_full_data.keys():
        key_np_full_data[key] = np.concatenate(tuple(key_np_data[key]), axis = None)
        print(len(key_np_full_data[key]))
        if max_length_data < len(key_np_full_data[key]):
            max_length_data = len(key_np_full_data[key])
    #print(key_np_full_back)

    for key in key_np_full_data.keys():                 
        #df_sig_full_np[key] = pd.Series(key_np_full[key])
        df_data_full_np[key] = list(np.append(key_np_full_data[key], np.repeat(np.nan, max_length_data- (len(key_np_full_data[key])))))
    #df_sig_full_np = pd.DataFrame([pd.Series(key_np_full[key]) for key in key_np_full.keys()], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
    print(df_data_full_np)
    df_dat_full_np = df_data_full_np[[f'{col}_{roi}' for col in names_sig_data]]
    df_dat_new_np = df_dat_full_np.dropna()
    print(df_dat_new_np)

    len_var = []
    for col in names_sig_data:
        len_var.append(len(df_dat_new_np[f'{col}_{roi}']))
        df_dat_new_np['target'] = np.full(np.max(len_var), 2, dtype = int)
    print(df_dat_new_np)
    df_data = pd.concat([df_data, df_dat_new_np], ignore_index = True)
df_data.to_csv(f'./plot/{folder_save}/numpy_data_DATA.csv', sep=',', encoding='utf-8', index=False)
######################################################################################

df = pd.concat([df_s_new_np, df_b_new_np], ignore_index = True)
print(df)
print(df.info())
df.to_csv(net_path + f'/xgb_training_dataset_{roi}.csv', sep=',', encoding='utf-8', index=False)

print("% of negative weights: " + str(len(df[f"wei_{roi}"][df[f"wei_{roi}"]<0])/len(df[f"wei_{roi}"])))

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)
names_sig_updated = ['m(H)', '$p_t$(H)', '$p_t$(Z)', 'm(Z)', '$p_t$($Z_{gen}$)', 'm($Z_{gen}$)', '$\\frac{p_t(V)}{p_t(H)}$', '$CvsL_{max}$',
                 '$CvsL_{min}$', '$CvsB_{max}$', '$CvsB_{min}$', '$p_t$ of $CvsL_{max}$ jet', '$p_t$ of $CvsL_{min}$ jet',
                 '$\Delta\Phi(V, H)$', '$\Delta R(jet_1, jet_2)$', '$\Delta\eta(jet_1, jet_2)$', '$\Delta\Phi(jet_1, jet_2)$', '$\Delta\Phi(l_1, l_2)$', '$\Delta\eta(l_1, l_2)$',
                 '$\Delta\Phi (l_{subleading}, jet_{subleading})$', '$\Delta\Phi (l_{subleading}, jet_{leading})$']

names_sig_updated_data = ['m(H)', '$p_t$(H)', '$p_t$(Z)', 'm(Z)', '$\\frac{p_t(V)}{p_t(H)}$', '$CvsL_{max}$',
                 '$CvsL_{min}$', '$CvsB_{max}$', '$CvsB_{min}$', '$p_t$ of $CvsL_{max}$ jet', '$p_t$ of $CvsL_{min}$ jet',
                 '$\Delta\Phi(V, H)$', '$\Delta R(jet_1, jet_2)$', '$\Delta\eta(jet_1, jet_2)$', '$\Delta\Phi(jet_1, jet_2)$', '$\Delta\Phi(l_1, l_2)$', '$\Delta\eta(l_1, l_2)$',
                 '$\Delta\Phi (l_{subleading}, jet_{subleading})$', '$\Delta\Phi (l_{subleading}, jet_{leading})$']  

c = 0

df_hists = pd.DataFrame([], columns = [f'{col}_{rois}' for col in names_sig for rois in roiis])
for col in names_sig_data[1:]:
    
    plt.figure(figsize=(10,10))
    len_sig = 0
    for i in range(0,len(df['target'])):
        if df['target'][i] == 1:
             len_sig += 1
    print(len_sig)
    names_big_ax = ['Higgs_mass', 'Higgs_pt', 'Z_pt', 'pt_lead', 'pt_sublead']
    if col in names_big_ax:
        hist.Hist.new.Regular(150, 0, 180).Double().fill(np.array(df[f'{col}_{roi}'][:len_sig])).plot()
        hist.Hist.new.Regular(150, 0, 180).Double().fill(np.array(df[f'{col}_{roi}'][len_sig:])).plot()
    else:
        hist.Hist.new.Regular(150, 0, 5).Double().fill(np.array(df[f'{col}_{roi}'][:len_sig])).plot()
        hist.Hist.new.Regular(150, 0, 5).Double().fill(np.array(df[f'{col}_{roi}'][len_sig:])).plot()
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
    plt.savefig(f"./plot/{folder_save}/{col}_{roi}.jpg")

    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80)

    data_counts, data_bins = np.histogram(np.array(df_data[f'{col}_{roi}']),bins =50, weights = np.array(df_data[f'wei_{roi}']))
    df_hists[f'{col}_{roi}'] = np.array(counts22)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 50, weights = np.array(df[f'wei_{roi}'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        density = True,
    )
    #
    #for i in range(0, len(bins2)-1):
    #    x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
    #    y_pos_sig = counts1[i] + (counts1[i] * 0.01)
    #    label_p_sig = str(counts11[i])
    #    x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
    #    y_pos = counts2[i] + (counts2[i] * 0.01)
    #    label_p = str(counts22[i])
    #    if i%5 == 0:
    #        ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
    #    if i%6 == 0:
    #        ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins = 50, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label= 'tt bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
    )
    ## plot compare list
    ax.errorbar(
        (data_bins[:-1] + data_bins[1:])/2,
        np.array(data_counts),
        label='Data',
        marker = 'o',
        color='k',
        yerr=np.sqrt(np.array(data_counts)),
        linestyle = "None",
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    sigratio = ratio * np.sqrt(np.where(counts1>0, (counts11/counts1**2) * 1/(np.sum(counts11))**(2) , 0) + np.where(counts2>0, (counts22/counts2**2) * 1/(np.sum(counts22))**(2), 0))
    plt.errorbar(bins1[:-1], ratio, yerr = np.abs(sigratio), color = "k", fmt = '.', marker = 'o', markeredgecolor = 'k')
    plt.plot(bins1[:-1], [1]*len(ratio), '--', color = 'black')
    
    
    ##  plot settings, adjust range
    rax.set_xlabel(f'{names_sig_updated_data[c]} {roi}')
    ax.set_xlabel(None)
    ax.set_ylabel("Events (normalised for sig/bg)")
    rax.set_ylabel('$\\frac{Signal}{Background}$')
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax, maxval, minval = autoranger(np.array(df[f'{col}_{roi}'][:len_sig]))
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
    fig.savefig(f"./plot/{folder_save}/compare_{col}_{roi}.pdf")
    fig.savefig(f"./plot/{folder_save}/compare_{col}_{roi}.jpg")
    
    ######################################################################################################
    #### Smaller scale ####################################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 160, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =160, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 160)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =160)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 160, weights = np.array(df[f'wei_{roi}'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        density = True,
    )
    for i in range(0, len(bins2)-1):
        x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
        y_pos_sig = counts1[i] + (counts1[i] * 0.01)
        label_p_sig = str(counts11[i])
        x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
        y_pos = counts2[i] + (counts2[i] * 0.01)
        label_p = str(counts22[i])
        if i%9 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%10 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =160, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 160, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =160, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    sigratio = ratio * np.sqrt(np.where(counts1>0, (counts11/counts1**2) * 1/(np.sum(counts11))**(2) , 0) + np.where(counts2>0, (counts22/counts2**2) * 1/(np.sum(counts22))**(2), 0))
    plt.errorbar(bins1[:-1], ratio, yerr = np.abs(sigratio), color = "k", fmt = '.', marker = 'o', markeredgecolor = 'k')
    plt.plot(bins1[:-1], [1]*len(ratio), '--', color = 'black')
    
    
    ##  plot settings, adjust range
    rax.set_xlabel(f'{names_sig_updated[c]} {roi}')
    ax.set_xlabel(None)
    ax.set_ylabel("Events (normalised)")
    rax.set_ylabel('$\\frac{Signal}{Background}$')
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax, maxval, minval = autoranger(np.array(df[f'{col}_{roi}'][:len_sig]))
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
    fig.savefig(f"./plot/{folder_save}/Small_scale/compare_{col}_{roi}.pdf")
    fig.savefig(f"./plot/{folder_save}/Small_scale/compare_{col}_{roi}.jpg")

    ######################################################################################################
    #### Larger scale #############################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 40, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =40, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 40)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =40)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 40, weights = np.array(df[f'wei_{roi}'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        density = True,
    )
    for i in range(0, len(bins2)-1):
        x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
        y_pos_sig = counts1[i] + (counts1[i] * 0.01)
        label_p_sig = str(counts11[i])
        x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
        y_pos = counts2[i] + (counts2[i] * 0.01)
        label_p = str(counts22[i])
        if i%4 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%5 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =40, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 40, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =40, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    sigratio = ratio * np.sqrt(np.where(counts1>0, (counts11/counts1**2) * 1/(np.sum(counts11))**(2) , 0) + np.where(counts2>0, (counts22/counts2**2) * 1/(np.sum(counts22))**(2), 0))
    plt.errorbar(bins1[:-1], ratio, yerr = np.abs(sigratio), color = "k", fmt = '.', marker = 'o', markeredgecolor = 'k')
    plt.plot(bins1[:-1], [1]*len(ratio), '--', color = 'black')
    
    
    ##  plot settings, adjust range
    rax.set_xlabel(f'{names_sig_updated[c]} {roi}')
    ax.set_xlabel(None)
    ax.set_ylabel("Events (normalised)")
    rax.set_ylabel('$\\frac{Signal}{Background}$')
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax, maxval, minval = autoranger(np.array(df[f'{col}_{roi}'][:len_sig]))
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
    fig.savefig(f"./plot/{folder_save}/Big_scale/compare_{col}_{roi}.pdf")
    fig.savefig(f"./plot/{folder_save}/Big_scale/compare_{col}_{roi}.jpg")

    ######################################################################################################
    #### Smaller scale but not that small ################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 120, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins = 120, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 120)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins = 120)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 120, weights = np.array(df[f'wei_{roi}'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        density = True,
    )
    for i in range(0, len(bins2)-1):
        x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
        y_pos_sig = counts1[i] + (counts1[i] * 0.01)
        label_p_sig = str(counts11[i])
        x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
        y_pos = counts2[i] + (counts2[i] * 0.01)
        label_p = str(counts22[i])
        if i%7 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%8 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins = 120, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 120, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins = 120, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    sigratio = ratio * np.sqrt(np.where(counts1>0, (counts11/counts1**2) * 1/(np.sum(counts11))**(2) , 0) + np.where(counts2>0, (counts22/counts2**2) * 1/(np.sum(counts22))**(2), 0))
    plt.errorbar(bins1[:-1], ratio, yerr = np.abs(sigratio), color = "k", fmt = '.', marker = 'o', markeredgecolor = 'k')
    plt.plot(bins1[:-1], [1]*len(ratio), '--', color = 'black')
    
    
    ##  plot settings, adjust range
    rax.set_xlabel(f'{names_sig_updated[c]} {roi}')
    ax.set_xlabel(None)
    ax.set_ylabel("Events (normalised)")
    rax.set_ylabel('$\\frac{Signal}{Background}$')
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))
    ax.legend()
    rax.set_ylim(0.0, 2.0)
    xmin, xmax, maxval, minval = autoranger(np.array(df[f'{col}_{roi}'][:len_sig]))
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
    fig.savefig(f"./plot/{folder_save}/Small_but_not_that_small_scale/compare_{col}_{roi}.pdf")
    fig.savefig(f"./plot/{folder_save}/Small_but_not_that_small_scale/compare_{col}_{roi}.jpg")
    
    c += 1
    
df_hists.to_csv(f'./plot/{folder_save}/hists_{roi}.csv', sep=',', encoding='utf-8', index=False)

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x-center)**2/(2*width**2)) + offset
  
def gaussiansin(x, height, center, width, offset, k, w):
    return height*np.exp(-(x-center)**2/(2*width**2)) + offset + k*np.sin(x*w)
  
def chiq2_gauss(x,y,sig,N,a):
    chiq1 = 0
    for i in range(0,N):
        chiq1 += ((y[i]-gaussian(x[i], a[0], a[1], a[2], a[3]))/sig[i])**2
    chiq1 = chiq1/(N-4)
    return chiq1
  
def chiq2_gausssin(x,y,sig,N,a):
    chiq1 = 0
    for i in range(0,N):
        chiq1 += ((y[i]-gaussiansin(x[i], a[0], a[1], a[2], a[3], a[4], a[5]))/sig[i])**2
    chiq1 = chiq1/(N-6)
    return chiq1

import scipy 
counts2, bins2 = np.histogram(np.array(df[f'del_phi_jj_{roi}'][len_sig:]),bins = 80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

counts22, bins22 = np.histogram(np.array(df[f'del_phi_jj_{roi}'][len_sig:]),bins = 80)

from scipy.fft import fft, fftfreq
from scipy import stats

yf = fft(counts22)

sampling_rate = 40

xf = fftfreq(sampling_rate*2, 1/ sampling_rate)

plt.figure(figsize = (13,8))
plt.plot(xf, np.abs(yf))
plt.savefig(f"./plot/{folder_save}/compare_FFT_{roi}.pdf")
plt.savefig(f"./plot/{folder_save}/compare_FFT_{roi}.jpg")

popt_s ,pcov_s = scipy.optimize.curve_fit(gaussiansin, bins22[:-1], counts22, sigma = np.sqrt(np.array(counts22)), absolute_sigma = True, p0= [100, 1.5, 0.5, 100, 1, 12])

popt_g ,pcov_g = scipy.optimize.curve_fit(gaussian, bins22[:-1], counts22, sigma = np.sqrt(np.array(counts22)), absolute_sigma = True, p0= [100, 1.5, 0.5, 100])

print("params gauss: ", popt_g)
print("params gauss + sin : ", popt_s)

print('\n Chi^2/dof of gauss sine is', chiq2_gausssin(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_s))
print('\n Chi^2 of gauss sine is', 6*chiq2_gausssin(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_s))

print('\n Chi^2/dof of gauss peak is', chiq2_gauss(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_g))
print('\n Chi^2 of gauss peak is', 4*chiq2_gauss(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_g))

p_val_sin = 1- stats.chi2.cdf(x=6*chiq2_gausssin(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_s), df=len(counts22)-6)
p_val_gauss = 1- stats.chi2.cdf(x=4*chiq2_gauss(bins22[:-1], counts22, np.sqrt(np.array(counts22)), len(bins22[:-1]), popt_g), df=len(counts22)-4)

print("p-value of gauss is: ", p_val_gauss, 1-p_val_gauss)
print("p-value of gauss + sin is: ", p_val_sin, 1- p_val_sin)
## plot compare list
def plot_data(x, y, unc, params, residuals, residuals_errors, pulls, pulls_errors, x_label, y_label, ylims, axes):
    xlin = np.linspace(0, 3.2)
    # Plot measurements and fitted parabola
    axes[0].errorbar(x, y, unc, linestyle='None', color='blue', fmt='.', label='DY bg')
    axes[0].plot(xlin, gaussian(xlin, *params), color='red', label='Fitted gaussian')
    axes[0].set_xlabel(x_label)
    axes[0].set_xlim(0, 3.2)
    axes[0].set_ylabel(y_label)
    axes[0].set_ylim(ylims[0], ylims[1])
    axes[0].legend()
    axes[0].grid(True)
    # Plot residuals
    axes[1].errorbar(x, residuals, yerr=residuals_errors, color='green', capsize=3, fmt='.', ls='')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Residuals')
    axes[1].grid(True)
    # Plot pulls
    axes[2].errorbar(x, pulls, yerr=pulls_errors, color='purple', capsize=3, fmt='.', ls='')
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel('Pulls')
    axes[2].grid(True)
    
def plot_data_sin(x, y, unc, params, residuals, residuals_errors, pulls, pulls_errors, x_label, y_label, ylims, axes):
    xlin = np.linspace(0, 3.2)
    # Plot measurements and fitted parabola
    axes[0].errorbar(x, y, unc, linestyle='None', color='blue', fmt='.', label='DY bg')
    axes[0].plot(xlin, gaussiansin(xlin, *params), color='red', label='Fitted gaussian + sin')
    axes[0].set_xlabel(x_label)
    axes[0].set_xlim(0, 3.2)
    axes[0].set_ylabel(y_label)
    axes[0].set_ylim(ylims[0], ylims[1])
    axes[0].legend()
    axes[0].grid(True)
    # Plot residuals
    axes[1].errorbar(x, residuals, yerr=residuals_errors, color='green', capsize=3, fmt='.', ls='')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Residuals')
    axes[1].grid(True)
    # Plot pulls
    axes[2].errorbar(x, pulls, yerr=pulls_errors, color='purple', capsize=3, fmt='.', ls='')
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel('Pulls')
    axes[2].grid(True)
    
error_count = np.sqrt(np.array(counts22))
res_gauss = np.array(counts22) - gaussian(bins22[:-1], *popt_g)
res_gauss_sin = np.array(counts22) - gaussiansin(bins22[:-1], *popt_s)

pulls_gauss = res_gauss/error_count
pulls_gauss_sin = res_gauss_sin/error_count
pulls_err_gauss = np.sqrt(error_count**2)/error_count

fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
yAxisRange = [0, 400]
# Plot the first column (existing data)
plot_data(bins22[:-1], counts22, error_count, popt_g, res_gauss, error_count, pulls_gauss, pulls_err_gauss, 'x', 'y', yAxisRange, axes[:, 0])
# Plot the second column (strange data)
plot_data_sin(bins22[:-1], counts22, error_count, popt_s, res_gauss_sin, error_count, pulls_gauss_sin, pulls_err_gauss, 'x', 'y (+sin)', yAxisRange, axes[:, 1])
# Adjust spacing between subplots
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0.3)
#plt.show()

fig.savefig(f"./plot/{folder_save}/compare_del_phi_jj_chi_{roi}.pdf")
fig.savefig(f"./plot/{folder_save}/compare_del_phi_jj_chi_{roi}.jpg")

X = df.drop("target", axis = 1)
print(X)
X = X.drop(f"wei_{roi}", axis = 1)
X = X.drop(f"Z_mass_{roi}", axis = 1)
X = X.drop(f"Z_pt_gen_{roi}", axis = 1)
X = X.drop(f"Z_mass_gen_{roi}", axis = 1)
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
plt.savefig(f"./plot/{folder_save}/importance.jpg")

plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(f"./plot/{folder_save}/boost_tree.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()
