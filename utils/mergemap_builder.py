# Goal: map the different names and samples in a json to merge them in histograms later
import os
import json
import argparse
parser = argparse.ArgumentParser(description='Build json map between different components of histogram stacks \
                                              and their respecitve names of samples that are contained within. \
                                              Note: Should be done before plotting histograms, and preferably also before running workflows.')
parser.add_argument('-c', '--channel', default=r'Zll', help='Channel ( Zll | Wln | Znn ) (default: %(default)s)')
parser.add_argument('-y', '--year', default=r'2017', help='Read sample information from AnalysisTools for this year (default: %(default)s)')
args = parser.parse_args()



if args.year == '2016':
    from samples_2016_FlavSplit_vhcc import get_samples_2016 as get_samples
elif args.year == '2017':
    from samples_2017_FlavSplit_vhcc import get_samples_2017 as get_samples
elif args.year == '2018':
    from samples_2018_FlavSplit_vhcc import get_samples_2018 as get_samples
    



samples_dict = get_samples(args.channel, signal_overlay=True, VJetsNLO=True)

stacks = []
stacks_dict = {}
for key in samples_dict:
    is_contained_in = samples_dict[key][2]
    if is_contained_in not in stacks:
        stacks.append(is_contained_in)
        stacks_dict[is_contained_in] = []

for stack in stacks:
    for key in samples_dict:
        if samples_dict[key][2] == stack and key != 'Data':
            stacks_dict[stack].append(key)
stacks_dict['Data'] = samples_dict['Data'][3]

print(stacks_dict)
with open('../metadata/mergemap_'+args.year+'_'+args.channel+'.json', 'w') as f:
    json.dump(stacks_dict, f, indent=4)
