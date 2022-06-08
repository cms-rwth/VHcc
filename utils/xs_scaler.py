import os
import sys

def read_xs(files):
    import json

    f = open(files)
    data = json.load(f)
    xs_dict={}
    for obj in data:
        #print(obj)
        xs_dict[obj]=float(data[obj]['xsec'])
    return xs_dict

def scale_xs(hist,lumi,events,unscale=False,year=2017,xsfile="metadata/sample_info_"):
    xs_dict = read_xs(os.getcwd()+"/"+xsfile+str(year)+'_reversed.json')
    #print(xs_dict)
    scales={}

    for key in events:
        #print(key)
        #continue
        #key_stripped = key.split('/')[1].split('/')[0]
        key_stripped = key
        print(key_stripped)
        if type(key) != str or key=="Data" or key not in xs_dict:
            continue
        if unscale: 
            scales[key]=events[key]/(xs_dict[key_stripped]*lumi)
        else :
            scales[key]=xs_dict[key_stripped]*lumi/events[key]
        #print(scales[key],key)
    hist.scale(scales, axis="dataset")
    return hist

def collate(accumulator, mergemap):
    out = {}
    for group, names in mergemap.items():
        out[group] = processor.accumulate([v for k, v in accumulator.items() if k in names])
    return out