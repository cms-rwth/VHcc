# Local Variables:
# python-indent-offset: 4
# End:

from VHcc.workflows.Zll_process_newHist_pandas_small_update_isolation import (
    NanoProcessor as VH_Zll,
)

cfg = {
    "userconfig": {'version':'test_nolepsf'},
    "dataset": {
        "jsons": [
                  "src/VHcc/metadata/bg_rwth_test_new_10.json"],
        "campaign": "UL17",
        "year": "2017",
        "filter": {
            "samples": [
                "DYJetsToLL_nlo_vau_bg"
                #"ZHToCC_vau_sig"
                #"DYJetsToLL_nlo",
                #"DY1ToLL_PtZ-250To400",
                #"DY1ToLL_PtZ-50To150",
                #"DY1ToLL_PtZ-150To250",
                #"DY1ToLL_PtZ-400ToInf",
                #"DY2ToLL_PtZ-50To150",
                #"DY2ToLL_PtZ-150To250",
                #"DY2ToLL_PtZ-250To400",
                #"DY2ToLL_PtZ-400ToInf",
                #"",
                #"",
                #"",
            ],
            "samples_exclude": [],
        },
    },
    # Input and output files
    "workflow": VH_Zll,
    "output": "output_vhcc_zll",
    "run_options": {
        #"executor": "parsl/condor/naf_lite",
        "executor": "parsl/condor",
        #"executor": "futures",
        "workers": 1,
        "scaleout": 150,
        "walltime": "01:00:00",
        "mem_per_worker": 2,  # GB
        "chunk": 50000,
        "max": None,
        "skipbadfiles": True,
        "voms": None,
        "limit": 80,
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
            '( Machine != "lxcip59.physik.rwth-aachen.de")'),
    },
}
