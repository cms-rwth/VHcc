# VHcc
Search for VH(cc) process with CMS data, using coffea processor

Follow [this link]( https://codimd.web.cern.ch/wLJlIq8jQtqJ-y7fP-kgdw#
) for setup instructions.

### Example job submission for VHcc
```
python runner.py --wf Zll --output Zll_vjets_17.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --workers 6 --scaleout 6
python runner.py --wf Zll --output Zll_vjets_ext_17.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_ext_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --workers 6 --scaleout 6
python runner.py --wf Zll --output Zll_other_17.coffea --json src/VHcc/metadata/mcsamples_2017_other_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --workers 6 --scaleout 6
python runner.py --wf Zll --output Zll_higgs_17.coffea --json src/VHcc/metadata/mcsamples_2017_higgs_Zll_used.json --executor parsl/condor/naf_lite --workers 6 --scaleout 6
python runner.py --wf Zll --output Zll_data_17.coffea --json src/VHcc/metadata/datasamples_2017_Zll_used.json --executor parsl/condor/naf_lite --workers 6 --scaleout 6
```

Trying more jobs, splitting into actual processor & merging
```
python runner.py --wf Zll --output Zll_vjets_17_just10.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite_merges --limit 10
python runner.py --wf Zll --output Zll_vjets_17_DYJetsToLL_nlo.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DYJetsToLL_nlo
python runner.py --wf Zll --output Zll_vjets_17_DY1ToLL_PtZ-50To150.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY1ToLL_PtZ-50To150
python runner.py --wf Zll --output Zll_vjets_17_DY1ToLL_PtZ-150To250.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY1ToLL_PtZ-150To250
python runner.py --wf Zll --output Zll_vjets_17_DY1ToLL_PtZ-250To400.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY1ToLL_PtZ-250To400
python runner.py --wf Zll --output Zll_vjets_17_DY1ToLL_PtZ-400ToInf.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY1ToLL_PtZ-400ToInf
python runner.py --wf Zll --output Zll_vjets_17_DY2ToLL_PtZ-50To150.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY2ToLL_PtZ-50To150
python runner.py --wf Zll --output Zll_vjets_17_DY2ToLL_PtZ-150To250.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY2ToLL_PtZ-150To250
python runner.py --wf Zll --output Zll_vjets_17_DY2ToLL_PtZ-250To400.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY2ToLL_PtZ-250To400
python runner.py --wf Zll --output Zll_vjets_17_DY2ToLL_PtZ-400ToInf.coffea --json src/VHcc/metadata/mcsamples_2017_vjets_Zll_used_nonCorruptedOnly.json --executor parsl/condor/naf_lite --only DY2ToLL_PtZ-400ToInf
```
### Useful commands to remove jobs from condor
#### Why?
To not fill the queue with more than the allowed number of individual jobs. (5000 I think)

(Because parsl does not delete finished jobs from the queue!)

#### How?
Remove successfully finished job (changes status to removed)
```
condor_rm --constraint 'JobStatus==4'
```
Actually force deletion of removed jobs from queue after being removed
```
condor_rm -forcex --constraint 'JobStatus==3'
```
If there are other jobs (unrelated) that should not be touched by this, there is also the option to remove a range of jobs.
```
condor_rm -forcex --constraint 'ClusterId > 21849531 && ClusterId < 21849551'
```