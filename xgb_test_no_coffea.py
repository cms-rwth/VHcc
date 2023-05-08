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
folder_save = 'eval_23_04_19_later'
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
names_sig = ['wei', 'Higgs_mass', 'Higgs_pt', 'Z_pt', 'Z_mass', 'Z_pt_gen', 'Z_mass_gen', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_eta_jj', 'del_phi_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading'] 

roiis = ['high_mumu', 'high_ee', 'low_mumu', 'low_ee']
roi = 'low_mumu'
######################################################################################
##### Read np arrays of signal sample ################################################
######################################################################################
paths_np = [str(x) for x in Path("./condor_signal_04_mid/ZHToCC_vau_sig").glob("**/*.npy") if ("_full" in str(x))] 
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
#print(key_np)
for key in key_np.keys():
    key_np[key] = [np.load(element) for element in key_np[key]]
#print(key_np)

key_np_full = {}
for col in names_sig:
    for rois in roiis:
        key_np_full[f'{col}_{rois}'] = np.array([])
for key in key_np_full.keys():
    key_np_full[key] = np.concatenate(tuple(key_np[key]), axis = None)
#print(key_np_full)

for key in key_np_full.keys():                 
    df_sig_full_np[key] = pd.Series(key_np_full[key])
print(df_sig_full_np)
df_s_new_np = df_sig_full_np[[f'{col}_{roi}' for col in names_sig]]
df_s_new_np = df_s_new_np.dropna()
print(df_s_new_np)
len_var = []
for col in names_sig:
    len_var.append(len(df_s_new_np[f'{col}_{roi}']))
    df_s_new_np['target'] = np.ones(np.max(len_var))
print(df_s_new_np)
######################################################################################


######################################################################################
##### Read np arrays of background sample ############################################
######################################################################################
paths_np_back = [str(x) for x in Path("./condor_back_04_mid/DYJetsToLL_nlo_vau_bg").glob("**/*.npy") if ("_full" in str(x))] 
#print(paths_np_back)
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
    key_np_back[key] = [np.load(element) for element in key_np_back[key]]
#print(key_np_back)

key_np_full_back = {}
for col in names_sig:
    for rois in roiis:
        key_np_full_back[f'{col}_{rois}'] = np.array([])
for key in key_np_full_back.keys():
    key_np_full_back[key] = np.concatenate(tuple(key_np_back[key]), axis = None)
#print(key_np_full_back)

for key in key_np_full_back.keys():                 
    df_back_full_np[key] = pd.Series(key_np_full_back[key])
print(df_back_full_np)
df_b_new_np = df_back_full_np[[f'{col}_{roi}' for col in names_sig]]
df_b_new_np = df_b_new_np.dropna()
print(df_b_new_np)

len_var = []
for col in names_sig:
    len_var.append(len(df_b_new_np[f'{col}_{roi}']))
    df_b_new_np['target'] = np.zeros(np.max(len_var))
print(df_b_new_np)

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
    plt.savefig(net_path +f"plot/{folder_save}/{col}_{roi}.jpg")



    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig])),
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
        if i%5 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%6 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    plt.plot(bins1[:-1], ratio, 'ko')
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
    fig.savefig(net_path +f"plot/{folder_save}/compare_{col}_{roi}.pdf")
    fig.savefig(net_path +f"plot/{folder_save}/compare_{col}_{roi}.jpg")
    
    ######################################################################################################
    #### No rescaling ####################################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]))
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]))

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig])),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        
    )
    for i in range(0, len(bins2)-1):
        x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
        y_pos_sig = counts1[i] + (counts1[i] * 0.01)
        label_p_sig = str(counts11[i])
        x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
        y_pos = counts2[i] + (counts2[i] * 0.01)
        label_p = str(counts22[i])
        if i%5 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%6 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:])),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    plt.plot(bins1[:-1], ratio, 'ko')
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
    fig.savefig(net_path +f"plot/{folder_save}/compare_no_dense_{col}_{roi}.pdf")
    fig.savefig(net_path +f"plot/{folder_save}/compare_no_dense_{col}_{roi}.jpg")

    ######################################################################################################
    #### No rescaling  hist density ######################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, density = True)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, density = True)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True),
        label= 'Higgs -> cc',
        histtype="step",
        color='r',
        yerr=True,
        ax=ax,
        
    )
    for i in range(0, len(bins2)-1):
        x_pos_sig = (bins1[i +1] - bins1[i])/4 + bins1[i]
        y_pos_sig = counts1[i] + (counts1[i] * 0.01)
        label_p_sig = str(counts11[i])
        x_pos = (bins2[i +1] - bins2[i])/4 + bins2[i]
        y_pos = counts2[i] + (counts2[i] * 0.01)
        label_p = str(counts22[i])
        if i%5 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%6 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    plt.plot(bins1[:-1], ratio, 'ko')
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
    fig.savefig(net_path +f"plot/{folder_save}/compare_np_dense_{col}_{roi}.pdf")
    fig.savefig(net_path +f"plot/{folder_save}/compare_np_dense_{col}_{roi}.jpg")

    ######################################################################################################
    #### No rescaling  hist density True #################################################################
    ######################################################################################################
    fig, ((ax), (rax)) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.cms.label("Private work", com="13.6", data=True, loc=0, ax=ax)
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)

    counts11, bins11 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, density = True)
    counts22, bins22 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, density = True)
    ## plot reference
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True),
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
        if i%5 == 0:
            ax.text(x_pos, y_pos, label_p, rotation = 'vertical', color = 'green')
        if i%6 == 0:
            ax.text(x_pos_sig, y_pos_sig, label_p_sig, rotation = 'vertical', color = 'red')
    ## plot compare list
    hep.histplot(
        np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True),
        label='DY bg',
        histtype="step",
        color='g',
        yerr=True,
        ax=ax,
        density = True,
        )
    # plot ratio of com/Ref
    
    counts1, bins1 = np.histogram(np.array(df[f'{col}_{roi}'][:len_sig]),bins = 80, weights = np.array(df[f'wei_{roi}'][:len_sig]), density = True)
    counts2, bins2 = np.histogram(np.array(df[f'{col}_{roi}'][len_sig:]),bins =80, weights = np.array(df[f'wei_{roi}'][len_sig:]), density = True)
    ratio = np.divide(counts1, counts2, where = (counts2 != 0))
    plt.plot(bins1[:-1], ratio, 'ko')
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
    fig.savefig(net_path +f"plot/{folder_save}/compare_np_dense_true_{col}_{roi}.pdf")
    fig.savefig(net_path +f"plot/{folder_save}/compare_np_dense_true_{col}_{roi}.jpg")
    
    c += 1

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
plt.savefig(net_path + f"plot/{folder_save}/importance.jpg")

plt.figure(figsize=(17,12))
plot_tree(xgb_cl, fmap = 'feature_map.txt')
plt.title('Decision tree graph')
#plt.show()
plt.savefig(net_path + f"plot/{folder_save}/boost_tree.jpg", dpi = 1800)
###result = 1/(1+np.exp(leaf_value))) for belonging to calss 1
#plt.show()
