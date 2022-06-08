# Use like:
# python fetch.py -i ../samples/mcsamples_2017_higgs_used.txt -s phys03 -o mcsamples_2017_higgs_used
# python fetch.py -i ../samples/mcsamples_2017_higgs_Wln_used.txt -s phys03 -o mcsamples_2017_higgs_Wln_used
# python fetch.py -i ../samples/mcsamples_2017_higgs_Zll_used.txt -s phys03 -o mcsamples_2017_higgs_Zll_used
# python fetch.py -i ../samples/mcsamples_2017_higgs_Znn_used.txt -s phys03 -o mcsamples_2017_higgs_Znn_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_used.txt -s phys03 -o mcsamples_2017_vjets_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_Wln_used.txt -s phys03 -o mcsamples_2017_vjets_Wln_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_Zll_used.txt -s phys03 -o mcsamples_2017_vjets_Zll_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_Znn_used.txt -s phys03 -o mcsamples_2017_vjets_Znn_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_ext_used.txt -s phys03 -o mcsamples_2017_vjets_ext_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_ext_Wln_used.txt -s phys03 -o mcsamples_2017_vjets_ext_Wln_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_ext_Zll_used.txt -s phys03 -o mcsamples_2017_vjets_ext_Zll_used
# python fetch.py -i ../samples/mcsamples_2017_vjets_ext_Znn_used.txt -s phys03 -o mcsamples_2017_vjets_ext_Znn_used
# python fetch.py -i ../samples/mcsamples_2017_other_used.txt -s phys03 -o mcsamples_2017_other_used
# python fetch.py -i ../samples/mcsamples_2017_other_Wln_used.txt -s phys03 -o mcsamples_2017_other_Wln_used
# python fetch.py -i ../samples/mcsamples_2017_other_Zll_used.txt -s phys03 -o mcsamples_2017_other_Zll_used
# python fetch.py -i ../samples/mcsamples_2017_other_Znn_used.txt -s phys03 -o mcsamples_2017_other_Znn_used
# python fetch.py -i ../samples/datasamples_2017_used.txt -s phys03 -o datasamples_2017_used
# python fetch.py -i ../samples/datasamples_2017_Wln_used.txt -s phys03 -o datasamples_2017_Wln_used
# python fetch.py -i ../samples/datasamples_2017_Zll_used.txt -s phys03 -o datasamples_2017_Zll_used
# python fetch.py -i ../samples/datasamples_2017_Znn_used.txt -s phys03 -o datasamples_2017_Znn_used
import os, sys
import json
import argparse
parser = argparse.ArgumentParser(description='Create json with individual paths for specified datasets.\
                                              Note: You should compare the available datasets with the necessary ones first to avoid running on unused samples.')
parser.add_argument('-i', '--input', default=r'singlemuon', help='List of samples in DAS (default: %(default)s)')
parser.add_argument('-s', '--site', default=r'global', help='Site (default: %(default)s)')
parser.add_argument('-o', '--output', default=r'singlemuon', help='Site (default: %(default)s)')
args = parser.parse_args()
fset = []

with open(args.input) as fp: 
    lines = fp.readlines() 
    for line in lines:
        if line[0] != '\n' and line[0] != '#':
            fset.append(line)

fdict = {}

year = '2017' if '2017' in args.input else ('2016' if '2016' in args.input else '2018')
with open(f'../metadata/sample_info_{year}.json') as si:
    info = json.load(si)
    info_dict={}
    for obj in info:
        #print(obj)
        info_dict[obj]=info[obj]['name']


instance = 'prod/'+args.site


xrd = 'root://xrootd-cms.infn.it//'

for dataset in fset:
    print(dataset)
    flist = os.popen(("/cvmfs/cms.cern.ch/common/dasgoclient -query='instance={} file dataset={}'").format(instance,fset[fset.index(dataset)].rstrip())).read().split('\n')
    dictname = dataset.rstrip()
    dictname = dictname[1:].split('/')[0]
    if dictname in ['SingleMuon','DoubleMuon','SingleElectron','DoubleElectron','DoubleEG','EGamma','MET']:
        dictname = dataset.rstrip()
    else:
        dictname = info_dict[dictname]
    if dictname not in fdict:
        fdict[dictname] = [xrd+f for f in flist if len(f) > 1]
    else: #needed to collect all data samples into one common key "Data" (using append() would introduce a new element for the key)
        fdict[dictname].extend([xrd+f for f in flist if len(f) > 1])

#print(fdict)
#pprint.pprint(fdict, depth=1)

with open('../metadata/%s.json'%(args.output), 'w') as fp:
    json.dump(fdict, fp, indent=4)
