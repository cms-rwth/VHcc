import os
import json
import argparse
parser = argparse.ArgumentParser(description='Convert sample information from AnalysisTools to json format. \
                                              Note: Reverse name and dir for easier lookup during workflow.')
parser.add_argument('-y', '--year', default=r'2017', help='Read sample information from AnalysisTools for this year (default: %(default)s)')
args = parser.parse_args()
fset = []
samples = '../metadata/sample_info_' + args.year
with open(samples+'.txt') as fp: 
    lines = fp.readlines() 
    for line in lines: 
        fset.append(line)

fdict = {}


for dataset in fset:
    #print(fset)
    if ('dir=' not in dataset) or (dataset[0] == '#'):
        #print(dataset)
        continue
    else:
        dataset = dataset.split('\n')[0]
        #print(fset)
        name = (dataset.split('name=')[1]).split(' ')[0]
        dir_ = (dataset.split('dir=')[1]).split(' ')[0]
        dir_ = dir_ if dir_[-1] != '/' else dir_[:-1]
        type_ = (dataset.split('type=')[1]).split(' ')[0]
        xsec = (dataset.split('xsec=')[1]).split(' ')[0]
        lepflav = (dataset.split('lepflav=')[1]).split(' ')[0] if 'lepflav=' in dataset else ''
        kfac = (dataset.split('kfac=')[1]).split(' ')[0] if 'kfac=' in dataset else ''
        doJetFlavorSplit = (dataset.split('doJetFlavorSplit=')[1]).split(' ')[0] if 'doJetFlavorSplit=' in dataset else ''
        npro = (dataset.split('npro=')[1]).split(' ')[0] if 'npro=' in dataset else ''
        
        print(name, dir_, type_, xsec, lepflav, kfac, doJetFlavorSplit, npro)
        
        fdict[name] = {
            'dir' : dir_,
            'type' : type_,
            'xsec' : xsec,
            'lepflav' : lepflav,
            'kfac' : kfac,
            'doJetFlavorSplit' : doJetFlavorSplit,
            'npro' : npro,
            }

with open(samples+'_reversed.json', 'w') as fp:
    json.dump(fdict, fp, indent=4)
