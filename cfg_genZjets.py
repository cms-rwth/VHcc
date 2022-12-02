# Local Variables:
# python-indent-offset: 4
# End:

from VHcc.workflows.genZjets import (
    NanoProcessor as zjets,
)


cfg = {
    "dataset": {
        "jsons": ["src/VHcc/metadata/genZjets.json"],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples": [
                "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
                "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                "DYJetsToEE_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                "DYJetsToMuMu_BornSuppressV3_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
            ],
            "samples_exclude": [],
        },
    },
    # Input and output files
    "workflow": zjets,
    "output": "output_zjets",
    "run_options": {
        "executor": "parsl/condor",
        #"executor": "futures",
        "workers": 1,
        "scaleout": 10,
        "walltime": "03:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 500000,
        "max": None,
        "skipbadfiles": None,
        "voms": None,
        "limit": 3,
        "retries": 20,
        "splitjobs": False,
    },
}
