# Local Variables:
# python-indent-offset: 4
# End:

from VHcc.workflows.Zll_process_newHist import (
    NanoProcessor as VH_Zll_newHist,
)
from VHcc.workflows.Zll_process import (
    NanoProcessor as VH_Zll,
)

cfg = {
    "userconfig": {'version':'test_nolepsf'},
    "dataset": {
        "jsons": [
            "src/VHcc/metadata/genZjets.json"
        ],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples": [
                "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
                #"DoubleMuon_Run2017E",
                #"DoubleMuon_Run2017F",
                #"DYJetsToLL_nlo",
                #"ZH125ToCC_ZLL_powheg",
                #"DY1ToLL_PtZ-250To400",
                #"DY1ToLL_PtZ-50To150",
                #"DY1ToLL_PtZ-150To250",
                #"DY1ToLL_PtZ-400ToInf",
                #"DY2ToLL_PtZ-50To150",
                #"DY2ToLL_PtZ-150To250",
                #"DY2ToLL_PtZ-250To400",
                #"DY2ToLL_PtZ-400ToInf",
                #"",
            ],
            "samples_exclude": [],
        },
    },
    # Input and output files
    "workflow": VH_Zll_newHist,
    "output": "output_vhcc_newHist_zll",
    "run_options": {
        #"executor": "parsl/condor/naf_lite",
        "executor": "futures",
        "workers": 10,
        "scaleout": 20,
        "walltime": "03:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 50000,
        "max": None,
        "skipbadfiles": True,
        "voms": None,
        "limit": 2,
        "retries": 20,
        "splitjobs": False,
    },
}
