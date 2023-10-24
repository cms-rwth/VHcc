# Local Variables:
# python-indent-offset: 4
# End:

from VHcc.workflows.recoZjets import (NanoProcessor as zjets,)

cfg = {
    "user": {"debug_level": 0,
             "cuts": {
                 "vpt": 0
             }
         },
    "dataset": {
        "jsons": [
            #"src/VHcc/metadata/run2UL18_files.json",
            "src/VHcc/metadata/run2UL18_files_Test.json",
            "src/VHcc/metadata/run2UL18_files_PoCo_SingleMuon.json",
            "src/VHcc/metadata/run2UL18_files_PoCo_EGamma.json",
        ],
        "campaign": "2018_UL",
        "year": "2018",
        "filter": {
            "samples": [
                "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
                "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
                #"DYJetsToMuMu_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                #"DYJetsToEE_M-50_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
                #"Run2UL18_DoubleMuon",
                #"Run2UL18_SingleMuon",
                #"Run2UL18_EGamma_GT36"
           ],

            "samples_exclude": [],
        },
    },

    "weights": {
        "common":{
            "inclusive":{
                "lumiMasks": "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
                "PU": None,
                "JME": "jec_compiled.pkl.gz",
                #"BTV": { "DeepJetC": "DeepJet_ctagSF_Summer20UL17_interp.root"},
                "LSF": {
                    "ele_ID 2018": "wp80iso",
                    "ele_Reco 2018": "RecoAbove20",
                    "ele_Reco_low 2018": "RecoBelow20",
                    ##"mvaEleID-Fall17-noIso-V2-wp80", ??
                    "mu_Reco 2018_UL": "NUM_TrackerMuons_DEN_genTracks",
                    #"mu_HLT 2017_UL": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
                    "mu_ID 2018_UL": "NUM_TightID_DEN_TrackerMuons",
                    "mu_Iso 2018_UL": "NUM_TightRelIso_DEN_TightIDandIPCut",
                    #"mu_ID_low NUM_TightID_DEN_TrackerMuons": "Efficiency_muon_trackerMuon_Run2017_UL_ID.histo.json",
                    #"mu_Reco_low NUM_TrackerMuons_DEN_genTracks": "Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.histo.json",
                },
            },
        },
    },
    "systematic": {
        "JERC": False,
        "weights": False,
    },
    "workflow": zjets,
    "output": "output/recoZjets_UL18",
    "run_options": {
        #"executor": "parsl/condor", "workers": 1,  "limit": None,
        "executor": "futures", "workers": 10,  "limit": 10,
        "scaleout": 350,
        "walltime": "01:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 500000,
        "max": None,
        "skipbadfiles": True,
        "voms": None,
        "retries": 20,
        "splitjobs": False,
        "requirements": (
            '( Machine != "lx1b02.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a03.physik.rwth-aachen.de") && '
            '( Machine != "lx3a05.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a06.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a09.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a13.physik.rwth-aachen.de") && '
            '( Machine != "lx3a14.physik.rwth-aachen.de") && '
            '( Machine != "lx3a15.physik.rwth-aachen.de") && '
            '( Machine != "lx3a23.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a25.physik.rwth-aachen.de") && '
            '( Machine != "lx3a27.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a46.physik.rwth-aachen.de") && '
            '( Machine != "lx3a44.physik.rwth-aachen.de") && '
            '( Machine != "lx3a47.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a55.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3a56.physik.rwth-aachen.de") && '
            '( Machine != "lx3b08.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b09.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b13.physik.rwth-aachen.de") && '
            '( Machine != "lx3b18.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b24.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b29.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b32.physik.rwth-aachen.de") && '
            '( Machine != "lx3b33.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b34.physik.rwth-aachen.de") && '
            '( Machine != "lx3b41.physik.rwth-aachen.de") && '
            '( Machine != "lx3b46.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b47.physik.rwth-aachen.de") && '
            '( Machine != "lx3b48.physik.rwth-aachen.de") && '
            '( Machine != "lx3b49.physik.rwth-aachen.de") && '
            '( Machine != "lx3b52.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b55.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b57.physik.rwth-aachen.de") && '
            '( Machine != "lx3b62.physik.rwth-aachen.de") && '
            '( Machine != "lx3b66.physik.rwth-aachen.de") && '
            '( Machine != "lx3b68.physik.RWTH-Aachen.de") && '
            '( Machine != "lx3b69.physik.rwth-aachen.de") && '
            '( Machine != "lx3b70.physik.rwth-aachen.de") && '
            '( Machine != "lx3b71.physik.rwth-aachen.de") && '
            '( Machine != "lx3b99.physik.rwth-aachen.de") && '
            '( Machine != "lxblade01.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade02.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade03.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade04.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade05.physik.rwth-aachen.de") && '
            '( Machine != "lxblade06.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade07.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade08.physik.rwth-aachen.de") && '
            '( Machine != "lxblade09.physik.rwth-aachen.de") && '
            '( Machine != "lxblade10.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade11.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade12.physik.rwth-aachen.de") && '
            '( Machine != "lxblade13.physik.rwth-aachen.de") && '
            '( Machine != "lxblade14.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade15.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade16.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade17.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade18.physik.rwth-aachen.de") && '
            '( Machine != "lxblade19.physik.rwth-aachen.de") && '
            '( Machine != "lxblade20.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade21.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade22.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade23.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade24.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade25.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade26.physik.rwth-aachen.de") && '
            '( Machine != "lxblade27.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade28.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade29.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade30.physik.RWTH-Aachen.de") && '
            '( Machine != "lxblade31.physik.rwth-aachen.de") && '
            '( Machine != "lxblade32.physik.rwth-aachen.de") && '
            '( Machine != "lxcip01.physik.rwth-aachen.de") && '
            '( Machine != "lxcip02.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip05.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip06.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip09.physik.rwth-aachen.de") && '
            '( Machine != "lxcip10.physik.rwth-aachen.de") && '
            '( Machine != "lxcip11.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip12.physik.rwth-aachen.de") && '
            '( Machine != "lxcip14.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip15.physik.rwth-aachen.de") && '
            '( Machine != "lxcip16.physik.rwth-aachen.de") && '
            '( Machine != "lxcip17.physik.rwth-aachen.de") && '
            '( Machine != "lxcip18.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip19.physik.rwth-aachen.de") && '
            '( Machine != "lxcip24.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip25.physik.rwth-aachen.de") && '
            '( Machine != "lxcip26.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip27.physik.rwth-aachen.de") && '
            '( Machine != "lxcip28.physik.rwth-aachen.de") && '
            '( Machine != "lxcip29.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip30.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip31.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip32.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip34.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip35.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip50.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip51.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip52.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip53.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip54.physik.RWTH-Aachen.de") && '
            '( Machine != "lxcip55.physik.rwth-aachen.de") && '
            '( Machine != "lxcip56.physik.rwth-aachen.de") && '
            '( Machine != "lxcip57.physik.rwth-aachen.de") && '
            '( Machine != "lxcip58.physik.rwth-aachen.de") && '
            '( Machine != "lxcip59.physik.rwth-aachen.de")'            
        ),
    },
}
