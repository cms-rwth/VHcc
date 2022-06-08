# python sample_config_comparison.py -n mcsamples_2017_vjets

import os, sys
import json
import argparse
parser = argparse.ArgumentParser(description='Compare sample list with actually used samples in VHcc and create used list of samples. \
                                              Note: You should only fetch files via DAS afterwards. Do both before running the processor on unnecessary files.')
parser.add_argument('-n', '--name', default=r'mcsamples_2017_higgs', help='Name of the file containing samples (default: %(default)s)')
args = parser.parse_args()


fset = []
fset_Wln = []
fset_Zll = []
fset_Znn = []

for y in ['2016','2017','2018']:
    if y in args.name:
        year = y
samples = '../metadata/sample_info_' + year

def read_VHcc_dir(files):
    f = open(files)
    data = json.load(f)
    return data

VHcc_dirs = read_VHcc_dir(samples+'.json')




# Figure out which samples are relevant for which channel

if '2016' in args.name:
    from samples_2016_FlavSplit_vhcc import get_samples_2016 as get_samples
elif '2017' in args.name:
    from samples_2017_FlavSplit_vhcc import get_samples_2017 as get_samples
elif '2018' in args.name:
    from samples_2018_FlavSplit_vhcc import get_samples_2018 as get_samples
    
    
used_samples = {}
for ch in ['Wln','Zll','Znn']:
    samples_read_in = get_samples(ch, signal_overlay=True, VJetsNLO=True)
    flat_sample_names = []
    for key in samples_read_in:
        inner_list = samples_read_in[key][3]
        for name in inner_list:
            if name not in flat_sample_names:
                flat_sample_names.append(name)
    used_samples[ch] = flat_sample_names

print(used_samples)
#sys.exit()




# Check the full list of available samples of a given category against the configuration for VHcc,
# prepend all irrelevant ones with #

with open('../samples/'+args.name+'.txt') as fp: 
    lines = fp.readlines() 
    for line in lines:
        if line[0] == '\n':
            fset.append(line)
        elif (line.split('/')[1]).split('/')[0] in VHcc_dirs.keys():
            fset.append(line)
            this_name = VHcc_dirs[(line.split('/')[1]).split('/')[0]]['name']
            if this_name in used_samples['Wln']:
                fset_Wln.append(line)
            if this_name in used_samples['Zll']:
                fset_Zll.append(line)
            if this_name in used_samples['Znn']:
                fset_Znn.append(line)
        else:
            fset.append('#'+line)

with open('../samples/'+args.name+'_used.txt', 'w') as f:
    for line in fset:
        f.write(line)
        #f.write('\n')



# Write the samples for different channels into different outfiles
with open('../samples/'+args.name+'_Wln_used.txt', 'w') as f:
    for line in fset_Wln:
        f.write(line)
        #f.write('\n')
with open('../samples/'+args.name+'_Zll_used.txt', 'w') as f:
    for line in fset_Zll:
        f.write(line)
        #f.write('\n')
with open('../samples/'+args.name+'_Znn_used.txt', 'w') as f:
    for line in fset_Znn:
        f.write(line)
        #f.write('\n')