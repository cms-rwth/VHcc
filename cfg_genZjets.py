# Local Variables:
# python-indent-offset: 4
# End:

from VHcc.workflows.genZjets import (
    NanoProcessor as zjets,
)


cfg = {
    "user": {"debug_level": 0,
             "cuts": {
                 "vpt": 100
             }
         },
    "dataset": {
        "jsons": [
            "src/VHcc/metadata/genZjets.json",
            "src/VHcc/metadata/test_samples_local_DY_UNLOPS.json",
        ],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples": [
                "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
                "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                "DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                "DYJetsToEE_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                "DYJetsToMuMu_BornSuppressV3_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                #"DYJets_UNLOPS"
                # "DYjetstomumu_01234jets_Pt-0ToInf_13TeV-sherpa", # NanoV7 (LHE_VPT variables missing)
           ],
            "samples_exclude": [],
        },
    },
    # Input and output files
    "workflow": zjets,
    "output": "output_zjets",
    "run_options": {
        "executor": "parsl/condor", "workers": 1,
        #"executor": "futures", "workers": 10,
        "scaleout": 400,
        "walltime": "05:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 500000,
        "max": None,
        "skipbadfiles": None,
        "voms": None,
        "limit": None,
        "retries": 20,
        "splitjobs": False,
    },
}
