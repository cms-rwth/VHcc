import argparse, sys, os, arrow, glob, yaml
import numpy as np
import matplotlib.pyplot as plt, mplhep as hep
from matplotlib.offsetbox import AnchoredText

from coffea.util import load
import hist

time = arrow.now().format("YY_MM_DD")
plt.style.use(hep.style.ROOT)

from BTVNanoCommissioning.utils.xs_scaler import collate, additional_scale
from BTVNanoCommissioning.utils.plot_utils import (
    isin_dict,
    check_config,
    load_coffea,
    load_default,
    rebin_and_xlabel,
    plotratio,
    autoranger,
)

parser = argparse.ArgumentParser(description="make comparison for different campaigns")
parser.add_argument("--cfg", type=str, required=True, help="Configuration files")
parser.add_argument(
    "--debug", action="store_true", help="Run detailed checks of yaml file"
)
args = parser.parse_args()

# load config from yaml
with open(args.cfg, "r") as f:
    config = yaml.safe_load(f)
## create output dictionary
if not os.path.isdir(f"plot/{config['output']}_{time}/"):
    os.makedirs(f"plot/{config['output']}_{time}/")
if args.debug:
    check_config(config, False)
## load coffea files
output = load_coffea(config, config["scaleToLumi"])
## build up merge map
mergemap = {}
if not any(".coffea" in o for o in output.keys()):
    for merger in config["mergemap"].keys():
        mergemap[merger] = [
            m for m in output.keys() if m in config["mergemap"][merger]["list"]
        ]
else:
    for merger in config["mergemap"].keys():
        flist = []
        for f in output.keys():
            flist.extend(
                [m for m in output[f].keys() if m in config["mergemap"][merger]["list"]]
            )
        mergemap[merger] = flist
refname = list(config["reference"].keys())[0]
collated = collate(output, mergemap)
config = load_default(config, False)

#print(collated)
#print(collated.keys())

#for k in collated.keys():
#    print(collated[k].keys())

if config["scaleToLumi"]:
    lumi=config["lumi"] / 1000.0,
else:
    lumi = None
hist_type = "step"
if "label" in config.keys():
    label = config["label"]
else:
    label = "Preliminary"


h1 = collated["MiNNLO"]["dilep_pt"]
h2 = collated["MiNNLO_Zpt_offi"]["dilep_pt"]


print(h1)
print(h2)

#hep.histplot(h1[{'lepflav': sum}], yerr=False)

h1 = h1[{'lepflav': sum}]
h2 = h2[{'lepflav': sum}]

ax_edges = h1.axes[0].edges

#from histoprint import text_hist, print_hist
#print_hist(h1, title="Dilepton pT")
#print_hist(h2, title="Dilepton pT")

print(h1.view())
print(h2.view())
#print(h1.values())
#print(h2.values())

rel_err_1 = np.sqrt(h1.variances())/h1.values()
rel_err_2 = np.sqrt(h2.variances())/h2.values()


print(rel_err_1)
print(rel_err_2)


fig, ax = plt.subplots(figsize=(10, 8))
fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
hep.cms.label(label, lumi=None, year=None, com=config["com"], data=False, loc=0, ax=ax)

hep.histplot([rel_err_1, rel_err_2], ax_edges, label=["Nominal", "Z-pt weighted"], histtype="step",  yerr=False, ax=ax)
ax.legend(title="Stat uncertainty of\nPowheg MiNNLO\nZ+Jets sample", loc=9, title_fontsize=16)

ax.set_xlabel("$p_T(\ell\ell)$")
ax.set_ylabel("relative stat. uncertainty; $\sigma/N$")
ax.set_ylim(bottom=0)
#ax.set_yscale("log")
at = AnchoredText(
        config["inbox_text"],
        loc=2,
        frameon=False,
    )
ax.add_artist(at)

fig.savefig(f"staterror.png")
