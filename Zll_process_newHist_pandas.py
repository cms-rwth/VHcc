import csv
from curses import meta
from dataclasses import dataclass
import gzip
import pickle, os, sys, mplhep as hep, numpy as np
from select import select

import json

#from coffea import hist, processor # ToDo: move to the better hist
from coffea import processor # ToDo: move to the better hist
import hist
from hist import Hist
from coffea.nanoevents.methods import vector
import awkward as ak
from VHcc.utils.correction import jec,muSFs,eleSFs,init_corr
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights, PackedSelection
from functools import partial
# import numba
from VHcc.helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4

def empty_column_accumulator():
    #return processor.column_accumulator(np.array([],dtype=object))
    return processor.column_accumulator(np.array([],dtype=np.float64))
def array_accumulator():
    return processor.defaultdict_accumulator(empty_column_accumulator)

def mT(obj1,obj2):
    return np.sqrt(2.*obj1.pt*obj2.pt*(1.-np.cos(obj1.phi-obj2.phi)))
def flatten(ar): # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)
def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar

def read_json(path):
    f = open(path)
    data = json.load(f)
    return data

def dataset_name_to_number(dataset, year):
    samples_path = 'src/VHcc/metadata/sample_info_' + year + '_reversed'

    samples = read_json(samples_path+'.json')
    
    return samples[dataset]['type'], samples[dataset]['doJetFlavorSplit']

def dataset_categories(year):
    map_path = 'src/VHcc/metadata/mergemap_' + year + '_Zll'
    
    samples = read_json(map_path+'.json').values()
    all_datasets = [item for sublist in samples for item in sublist]
    
    return all_datasets

def get_info_dict(year):
    with open(f'src/VHcc/metadata/sample_info_{year}.json') as si:
        info = json.load(si)
        info_dict={}
        for obj in info:
            #print(obj)
            info_dict[obj]=info[obj]['name']
        return info_dict

class NanoProcessor(processor.ProcessorABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._campaign = self.cfg.dataset["campaign"]

        self._version=self.cfg.userconfig['version'] # only because the new runner etc. needs that, not used later
        self._export_array = True # if 'test' in self._version else False
        self._debug = False #True
        
        # paths from table 1 and 2 of the AN_2020_235
        
        # l l
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L3230-L3337
        self._mumu_hlt = {
            '2016': [
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ'
            ],
            '2017': [
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',#allowMissingBranch=1
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'#allowMissingBranch=1
            ],
            '2018': [
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',#allowMissingBranch=1 but this is the only used one in 2018?!
                #'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'#allowMissingBranch=1
            ],
        }   
    
        self._ee_hlt = {
            '2016': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'
            ],
            '2017': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                #'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ' # not in VHccAnalysis code
            ],
            '2018': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'
            ],
        }  
        
        '''
        # l nu
        self._munu_hlt = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24'
            ],
            '2017': [
                'IsoMu24',
                'IsoMu27'
            ],
            '2018': [
                'IsoMu24',
                'IsoMu27'
            ],
        }   
    
        self._enu_hlt = {
            '2016': [
                'Ele27_eta2p1_WPTight_Gsf'
            ],
            '2017': [
                'Ele32_WPTight_Gsf_L1DoubleEG',
                'Ele32_WPTight_Gsf'
            ],
            '2018': [
                'Ele32_WPTight_Gsf_L1DoubleEG',
                'Ele32_WPTight_Gsf'#allowMissingBranch=1
            ],
        }  
        
        # nu nu
        self._nunu_hlt = {
            '2016': [
                'PFMET110_PFMHT110_IDTight',
                #'PFMET110_PFMHT120_IDTight', # found in hltbranches_2016.txt but not in AN, maybe redundant?
                'PFMET170_NoiseCleaned',#allowMissingBranch=1
                'PFMET170_BeamHaloCleaned',#allowMissingBranch=1
                'PFMET170_HBHECleaned'
            ],
            '2017': [
                'PFMET110_PFMHT110_IDTight',
                'PFMET120_PFMHT120_IDTight',
                'PFMET120_PFMHT120_IDTight_PFHT60',#allowMissingBranch=1
                'PFMETTypeOne120_PFMHT120_IDTight'
            ],
            '2018': [
                'PFMET110_PFMHT110_IDTight',
                'PFMET120_PFMHT120_IDTight',
                'PFMET120_PFMHT120_IDTight_PFHT60'#allowMissingBranch=1
            ],
        } 
        
        '''
        
        # differences between UL and EOY
        # see https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        # also look at sec. 3.7.2
        self._met_filters = {
            '2016': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    'eeBadScFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'eeBadScFilter', # not suggested in EOY MC
                ],
            },
            '2017': {
                "data": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
                "mc": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
            },
            '2018': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    #'BadPFMuonDzFilter', # not in EOY
                    #'hfNoisyHitsFilter', # not in EOY
                    #'eeBadScFilter', # not suggested in EOY MC
                    'ecalBadCalibFilterV2',
                ],
            },
        }
        
        # https://gitlab.cern.ch/aachen-3a/vhcc-nano/-/blob/master/crab/crab_all.py#L33-36
        #'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions16/13TeV/ReReco/Final/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'
        #'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions17/13TeV/ReReco/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt' 
        #'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions18/13TeV/ReReco/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'
        # downloaded.
        self._lumiMasks = {
            '2016': LumiMask('src/VHcc/data/Lumimask/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'),
            '2017': LumiMask('src/VHcc/data/Lumimask/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt'),
            '2018': LumiMask('src/VHcc/data/Lumimask/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt')
        }
        
        self._corr = init_corr(self._year)
        
        # Axes: Cat - what it is, a type of something, described with words
        #       Bin - how much of something, numerical things
        #
        #   --> Some axes are already connected to specific objetcs, or to the event
        #   --> Others are "building-blocks" that can be reused multiple times
        
        list_of_datasets = dataset_categories(self._year)
        #print(list_of_datasets)
        #sys.exit()
        # Define axes
        # Should read axes from NanoAOD config / metadata
        #dataset_axis = hist.Cat("dataset", "Primary dataset")
        dataset_axis = hist.axis.StrCategory([],      name="dataset", label="Primary dataset", growth=True)
        # split V+jets sample & VZ signal, this is per event
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L2184-L2276
        #datasetSplit_axis = hist.Cat("datasetSplit", "Dataset split by flav", list_of_datasets)
        datasetSplit_axis = hist.axis.StrCategory(list_of_datasets, name="datasetSplit", label="Dataset split by flav")
        
        # use hadronFlavour, necessary when applying btag scale factors (that depend on flavour)
        # this one will be done per jet, can have values 0, 4, 5
        #flav_axis = hist.Bin("flav", r"hadronFlavour",[0,1,4,5,6])
        flav_axis = hist.axis.Variable([0,1,4,5,6], name="flav", label="hadronFlavour")
        
        #lepflav_axis = hist.Cat("lepflav",['ee','mumu'])
        lepflav_axis = hist.axis.StrCategory(['ee','mumu'], name="lepflav", label="Lepton flav")
        
        regions = ['SR_2LL','SR_2LH',
                   'CR_Zcc_2LL','CR_Zcc_2LH',
                   'CR_Z_LF_2LL','CR_Z_LF_2LH',
                   'CR_Z_HF_2LL','CR_Z_HF_2LH',
                   'CR_t_tbar_2LL','CR_t_tbar_2LH']
        #region_axis = hist.Cat("region",regions)
        region_axis = hist.axis.StrCategory(regions, name="region", label="Region")
        
        # Events
        njet_axis  = hist.axis.Regular(13, -.5, 12.5, name="nj", label="N jets") #hist.Bin("nj",  r"N jets", 13, -.5, 12.5)
        
        nAddJets_axis  = hist.axis.Regular(11, -.5, 10.5, name="nAddJets302p5_puid", label="N additional jets")
        #hist.Bin("nAddJets302p5_puid",  r"N additional jets", 11, -.5, 10.5)
        nAddJets_FSRsub_axis  = hist.axis.Regular(11, -.5, 10.5, name="nAddJetsFSRsub302p5_puid", label="N additional jets (FSR subtracted)")
        #hist.Bin("nAddJetsFSRsub302p5_puid",  r"N additional jets (FSR subtracted)", 11, -.5, 10.5)
        
        #nbjet_axis = hist.Bin("nbj", r"N b jets",    [0,1,2,3,4,5])            
        #ncjet_axis = hist.Bin("ncj", r"N c jets",    [0,1,2,3,4,5])
        # kinematic variables       
        pt_axis   = hist.axis.Regular(50, 0, 300, name="pt", label=r"$p_{T}$ [GeV]")
        #hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 300)
        eta_axis  = hist.axis.Regular(25, -2.5, 2.5, name="eta", label=r"$\eta$")
        #hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.axis.Regular(30, -3, 3, name="phi", label=r"$\phi$")
        #hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.axis.Regular(50, 0, 300, name="mass", label=r"$m$ [GeV]")
        #hist.Bin("mass", r" $m$ [GeV]", 50, 0, 300)
        mt_axis = hist.axis.Regular(30, 0, 300, name="mt", label=r"$m_{T}$ [GeV]")
        #hist.Bin("mt", r" $m_{T}$ [GeV]", 30, 0, 300)
        dr_axis = hist.axis.Regular(20, 0, 5, name="dr", label=r"$\Delta$R")
        #hist.Bin("dr","$\Delta$R",20,0,5)
        
        # some more variables to check, which enter BDT
        # need to revisit this later, because high Vpt and low Vpt can have different binning
        jjVPtRatio_axis = hist.axis.Regular(15, 0, 2, name="jjVPtRatio", label=r"$p_{T}(jj) / $p_{T}(V)$ [GeV]")
        #hist.Bin("jjVPtRatio",r"$p_{T}(jj) / $p_{T}(V)$ [GeV]",15,0,2)
        
        
        #dphi_V_H_axis = hist.Bin("dphi_V_H","$\Delta\Phi(V, H)$",20,0,3.2)
        # jet jet
        #dr_j1_j2_axis = hist.Bin("dr_j1_j2","$\Delta R(j1,j2)$",20,0,5)
        # jet jet
        #dphi_j1_j2_axis = hist.Bin("dphi_j1_j2","$\Delta\Phi(j1,j2)$",15,-3.2,3.2)
        #deta_j1_j2_axis = hist.Bin("deta_j1_j2","$\Delta\eta(j1,j2)$",15,0,3)
        # lepton lepton
        #dphi_l1_l2_axis = hist.Bin("dphi_l1_l2","$\Delta\Phi(l1,l2)$",15,0,3.2)
        #deta_l1_l2_axis = hist.Bin("eta_l1_l2","$\Delta\eta(l1,l2)$",15,0,2.6)
        # jet lepton
        #dphi_j1_l1_axis = hist.Bin("dphi_j1_l1","$\Delta\Phi(j1,l1)$",15,0,3.2)
        #dphi_j2_l1_axis = hist.Bin("dphi_j2_l1","$\Delta\Phi(j2,l1)$",15,0,3.2)
        #dphi_j1_l2_axis = hist.Bin("dphi_j1_l2","$\Delta\Phi(j1,l2)$",15,0,3.2)
        #dphi_j2_l2_axis = hist.Bin("dphi_j2_l2","$\Delta\Phi(j2,l2)$",15,0,3.2)
        
        # ToDo: several other variables can only be stored after kinfit
        # e.g. here https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/python/kinfitter.py
        # or https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L4670-L4995
        
        # weights are interesting as well
        weight_axis = hist.axis.Regular(100, -5.001, 5.001, name="weight_full", label="weight_full")
        #hist.Bin("weight_full","weight_full",100,-5.001, 5.001)
        genweight_axis = hist.axis.Regular(100, -0.001, 0.001, name="genWeight", label="genWeight")
        #hist.Bin("genWeight","genWeight",100,-0.001, 0.001)
        sign_genweight_axis = hist.axis.Regular(100, -1.001, 1.001, name="genWeight_by_abs", label="genWeight/abs(genWeight)")
        #hist.Bin("genWeight_by_abs","genWeight/abs(genWeight)",100,-1.001,1.001)
        
        
        # MET vars
        #signi_axis = hist.Bin("significance", r"MET $\sigma$",20,0,10)
        #covXX_axis = hist.Bin("covXX",r"MET covXX",20,0,10)
        #covXY_axis = hist.Bin("covXY",r"MET covXY",20,0,10)
        #covYY_axis = hist.Bin("covYY",r"MET covYY",20,0,10)
        #sumEt_axis = hist.Bin("sumEt", r" MET sumEt", 50, 0, 300)
        
        # ToDo: switch to this
        # axis.StrCategory([], name='region', growth=True),
        #disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        # As far as I can tell, we only need DeepFlav currently
        ### In all of the older stuff use:
        #disc_list = ['btagDeepFlavC','btagDeepFlavB','btagDeepFlavCvL','btagDeepFlavCvB']
        ### With new stuff UL I use, use:
        disc_list = ['btagDeepFlavCvL','btagDeepFlavCvB']
        btag_axes = []
        for d in disc_list:
            # technically, -1 values are possible, but probably unlikely to matter much after event selection
            btag_axes.append(hist.axis.Regular(20, 0, 1, name=d, label=d)
                #hist.Bin(d, d , 20, 0, 1)
            )  
        #h = (
        #    Hist.new.Reg(10, -5, 5, overflow=False, underflow=False, name="A")
        #    .Bool(name="B")
        #    .Var(range(10), name="C")
        #    .Int(-5, 5, overflow=False, underflow=False, name="D")
        #    .IntCat(range(10), name="E")
        #    .StrCat(["T", "F"], name="F")
        #    .Double()
        #)
        #print(type(dataset_axis))
        #print(type(lepflav_axis))
        #print(type(flav_axis))
        #print(type(njet_axis))
        #print(type(hist.storage.Weight()))
        #testHistA = Hist(dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, njet_axis, hist.storage.Weight())    
        #testHist = Hist(
        #            #dataset_axis,
        #            #datasetSplit_axis,
        #            #lepflav_axis,
        #            #region_axis,
        #            njet_axis,
        #            hist.storage.Weight()
        #        )    
        _hist_event_dict = {
                'nj'                       : Hist(datasetSplit_axis,
                                                       lepflav_axis,
                                                       region_axis,
                                                       njet_axis, hist.storage.Weight()),
                'nAddJets302p5_puid'       : Hist(datasetSplit_axis,
                                                       lepflav_axis,
                                                       region_axis,
                                                       nAddJets_axis, hist.storage.Weight()),
                'nAddJetsFSRsub302p5_puid' : Hist(datasetSplit_axis,
                                                       lepflav_axis,
                                                       region_axis,
                                                       nAddJets_FSRsub_axis, hist.storage.Weight()),
               # 'weight_full'              : Hist(datasetSplit_axis,
               #                                        lepflav_axis,
               #                                        region_axis,
               #                                        weight_axis, hist.storage.Weight()),
               # 'genweight'                : Hist(datasetSplit_axis,
               #                                        lepflav_axis,
               #                                        region_axis,
               #                                        genweight_axis, hist.storage.Weight()),
               # 'sign_genweight'           : Hist(datasetSplit_axis,
               #                                        lepflav_axis,
               #                                        region_axis,
               #                                        sign_genweight_axis, hist.storage.Weight()),
                'jjVPtRatio'               : Hist(datasetSplit_axis,
                                                       lepflav_axis,
                                                       region_axis,
                                                       jjVPtRatio_axis, hist.storage.Weight())
            
                #'dphi_V_H'                 : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, ,dphi_V_H_axis)
                #'dr_j1_j2'                 : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, ,dr_j1_j2_axis)
                #'dphi_j1_j2'               : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, dphi_j1_j2_axis)
                #'deta_j1_j2'               : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, deta_j1_j2_axis)
                #'dphi_l1_l2'               : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, dphi_l1_l2_axis)
                #'dphi_j1_l2'               : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, dphi_j1_l2_axis)
                #'dphi_j2_l2'               : Hist( dataset_axis, datasetSplit_axis, lepflav_axis, region_axis, dphi_j2_l2_axis)
                
                #'sampleFlavSplit'  : Hist( dataset_axis,  lepflav_axis, region_axis, sampleFlavSplit_axis),
                #'nbj' : Hist( dataset_axis, lepflav_axis, region_axis, nbjet_axis),
                #'ncj' : Hist( dataset_axis, lepflav_axis, region_axis, ncjet_axis),
                #'hj_dr'  : Hist( dataset_axis, lepflav_axis, region_axis, dr_axis),
                #'MET_sumEt' : Hist( dataset_axis, lepflav_axis, region_axis, sumEt_axis),
                #'MET_significance' : Hist( dataset_axis, lepflav_axis, region_axis, signi_axis),
                #'MET_covXX' : Hist( dataset_axis, lepflav_axis, region_axis, covXX_axis),
                #'MET_covXY' : Hist( dataset_axis, lepflav_axis, region_axis, covXY_axis),
                #'MET_covYY' : Hist( dataset_axis, lepflav_axis, region_axis, covYY_axis),
                #'MET_phi' : Hist( dataset_axis, lepflav_axis, region_axis, phi_axis),
                #'MET_pt' : Hist( dataset_axis, lepflav_axis, region_axis, pt_axis),
                #'mT1' : Hist( dataset_axis, lepflav_axis, region_axis, mt_axis),
                #'mT2' : Hist( dataset_axis, lepflav_axis, region_axis, mt_axis),
                #'mTh':Hist( dataset_axis, lepflav_axis, region_axis, mt_axis),
                #'dphi_lep1':Hist( dataset_axis, lepflav_axis, region_axis, phi_axis),
                #'dphi_lep2':Hist( dataset_axis, lepflav_axis, region_axis, phi_axis),
                #'dphi_ll':Hist( dataset_axis, lepflav_axis, region_axis, phi_axis),
            }
        
        # jets will be ordered by DeepJet (which is DeepFlav for historical reasons)
        objects=['leading_jetflav','subleading_jetflav','lep1','lep2','ll','jj']
        
        for i in objects:
            # distinguish between jets and other objects, as the structure for jets contains additional flavour axis
            if 'jet' in i: 
                _hist_event_dict["%s_pt" %(i)]=Hist(datasetSplit_axis,
                                                         lepflav_axis,
                                                         region_axis,
                                                         flav_axis,
                                                         pt_axis, hist.storage.Weight())
                _hist_event_dict["%s_eta" %(i)]=Hist(datasetSplit_axis,
                                                          lepflav_axis,
                                                          region_axis,
                                                          flav_axis,
                                                          eta_axis, hist.storage.Weight())
                _hist_event_dict["%s_phi" %(i)]=Hist(datasetSplit_axis,
                                                          lepflav_axis,
                                                          region_axis,
                                                          flav_axis,
                                                          phi_axis, hist.storage.Weight())
                _hist_event_dict["%s_mass" %(i)]=Hist(datasetSplit_axis,
                                                           lepflav_axis,
                                                           region_axis,
                                                           flav_axis,
                                                           mass_axis, hist.storage.Weight())
            else:
                _hist_event_dict["%s_pt" %(i)]=Hist(datasetSplit_axis,
                                                         lepflav_axis,
                                                         region_axis,
                                                         pt_axis, hist.storage.Weight())
                _hist_event_dict["%s_eta" %(i)]=Hist(datasetSplit_axis,
                                                          lepflav_axis,
                                                          region_axis,
                                                          eta_axis, hist.storage.Weight())
                _hist_event_dict["%s_phi" %(i)]=Hist(datasetSplit_axis,
                                                          lepflav_axis,
                                                          region_axis,
                                                          phi_axis, hist.storage.Weight())
                _hist_event_dict["%s_mass" %(i)]=Hist(datasetSplit_axis,
                                                           lepflav_axis,
                                                           region_axis,
                                                           mass_axis, hist.storage.Weight())
        
        # more information on the discriminators is stored for the first two jets,
        # ordered by DeepJet CvL discriminator and called "leading" and "subleading"
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["leading_jetflav_%s" %(disc)] = Hist(datasetSplit_axis,
                                                                       lepflav_axis,
                                                                       region_axis,
                                                                       flav_axis,
                                                                       axis, hist.storage.Weight())
            _hist_event_dict["subleading_jetflav_%s" %(disc)] = Hist(datasetSplit_axis,
                                                                          lepflav_axis,
                                                                          region_axis,
                                                                          flav_axis,
                                                                          axis, hist.storage.Weight())
            
        self.event_hists = list(_hist_event_dict.keys())
        
        # this can be used to not only store histograms, but also features on a per-event basis (arrays)
        if self._export_array:
            _hist_event_dict['array'] = processor.defaultdict_accumulator(array_accumulator)
        #self._accumulator = processor.dict_accumulator(
        #    {**_hist_event_dict,   
        #     #'cutflow': processor.defaultdict_accumulator(
        #     #    partial(processor.defaultdict_accumulator, int))
        #    })
        #self._accumulator['sumw'] = processor.defaultdict_accumulator(float)
        
        #self._accumulator = processor.dict_accumulator(
        #    {
        #        observable: Hist.Hist(var_axis, name="Counts", storage="Weight")
        #        for observable, var_axis in axis.items()
        #        if observable != "dataset"
        #    }
        #)
        #self._accumulator["cutflow"] = processor.defaultdict_accumulator(
        #    partial(processor.defaultdict_accumulator, int)
        #)
        #self._accumulator["sumw"] = 0
        
        self.make_output = lambda: {
            "cutflow": processor.defaultdict_accumulator(
                partial(processor.defaultdict_accumulator, int)
            ),
            "sumw": 0,
            **_hist_event_dict
        }


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        #output = self.accumulator #.identity()
        output = self.make_output()
        dataset = events.metadata['dataset']
        start = events.metadata['entrystart']
        stop = events.metadata['entrystop']
        output_location_list = []
        filename = events.metadata['filename'].split('/')[-1].strip('.root')
        #print(dataset)
        # Q: could there be MC that does not have this attribute? Or is it always the case?
        isRealData = not hasattr(events, "genWeight")
        
        # Done (externally): map from the lengthy dataset (path) to a more readable name
        # Keep the long name only for data, because it contains the Run info (necessary to apply corrections)
        if isRealData:
            info_dict = get_info_dict(self._year)
            dataset_long = dataset
            dictname = dataset[1:].split('/')[0]
            dataset = info_dict[dictname]
        print(dataset)    
        sample_type, doFlavSplit = dataset_name_to_number(dataset, self._year)
        # length of events is used so many times later on, probably useful to just save it here and then refer to that
        nEvents = len(events)
        print('Number of events: ', nEvents)
        
        # As far as I understand, this looks like a neat way to give selections a name,
        # while internally, there are boolean arrays for all events
        selection = PackedSelection()
        
        
        # this is either counting events in data with weight 1, or weighted (MC)
        if isRealData:
            output['sumw'] += nEvents
        else:
            # instead of taking the weights themselves, the sign is used:
            # https://cms-talk.web.cern.ch/t/huge-event-weights-in-dy-powhegminnlo/8718/7
            # although I initially had the same concerns as those raised in the thread,
            # if not only the sign is different, but also the absolute values between events
            # somehow it seems to average out, although I don't see why this is guaranteed
            # must have to do with "LO without interference" where the values are indeed same
            # and if they are not same, the differences are consired to be negligible
            output['sumw'] += ak.sum(events.genWeight/abs(events.genWeight))
            
            
        req_lumi=np.ones(nEvents, dtype='bool')
        if isRealData: 
            req_lumi=self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumi',ak.to_numpy(req_lumi))
        del req_lumi
        
        
        # AS: sort of the same thing as above, but now per entry
        weights = Weights(nEvents, storeIndividual=True)
        if isRealData:
            weights.add('genweight',np.ones(nEvents))
        else:
            weights.add('genweight',events.genWeight/abs(events.genWeight))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
            
            
        ##############
        if isRealData:
            output['cutflow'][dataset]['all']  += nEvents
            output['cutflow'][dataset]['all (weight 1)']  += nEvents
        else:
            output['cutflow'][dataset]['all']  += ak.sum(events.genWeight/abs(events.genWeight))
            output['cutflow'][dataset]['all (weight 1)']  += nEvents
            
        
        #trigger_met = np.zeros(nEvents, dtype='bool')

        trigger_ee = np.zeros(nEvents, dtype='bool')
        trigger_mm = np.zeros(nEvents, dtype='bool')

        #trigger_e = np.zeros(nEvents, dtype='bool')
        #trigger_m = np.zeros(nEvents, dtype='bool')
        
        #for t in self._nunu_hlt[self._year]:
        #    # so that already seems to be the check for whether the path exists in the file or not
        #    if t in events.HLT.fields:
        #        trigger_met = trigger_met | events.HLT[t]

        for t in self._mumu_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]

        for t in self._ee_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]

        #for t in self._munu_hlt[self._year]:
        #    if t in events.HLT.fields:
        #        trigger_m = trigger_m | events.HLT[t]

        #for t in self._emu_hlt[self._year]:
        #    if t in events.HLT.fields:
        #        trigger_e = trigger_e | events.HLT[t]
        
        
        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mm))
        
        
        # apart from the comments above about EOY/UL, should be fine
        metfilter = np.ones(nEvents, dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter
        
        
        
        # Not strictly necessary for Zll
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "energy": events.MET.sumEt,
                    }, with_name="PtEtaPhiMLorentzVector"
                )
        
        
        
        split_by_flav = False
        sampleFlavSplit = np.zeros(nEvents)
        possible_flavSplits = ['already_split_sample']
        selection.add('already_split_sample',sampleFlavSplit == 0)
        if not isRealData and not self._debug:
            if doFlavSplit == '1' and not (int(sample_type) >= 27 and int(sample_type) <= 39):
                split_by_flav = True
                # uses the same naming scheme as AT, although udbsg is counterintuitive (b? [sic!])
                possible_flavSplits = ['_cc','_bb','_bc','_cl','_bl','_udbsg']
                # =================================================================================
                #
                # #                       Split V+jets BG by flavour, via GenJet
                #
                # ---------------------------------------------------------------------------------
                # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L2184-L2228
                gen_jet = events.GenJet

                cGenJetTot = ak.sum((gen_jet.hadronFlavour == 4) & (gen_jet.pt > 20) & (abs(gen_jet.eta) < 2.4), axis=1)
                bGenJetTot = ak.sum((gen_jet.hadronFlavour == 5) & (gen_jet.pt > 20) & (abs(gen_jet.eta) < 2.4), axis=1)

                tag_cc = cGenJetTot >= 2
                tag_bb = bGenJetTot >= 2
                tag_bc = (bGenJetTot == 1) & (cGenJetTot == 1)
                tag_cl = (cGenJetTot == 1) & (bGenJetTot == 0)
                tag_bl = (bGenJetTot == 1) & (cGenJetTot == 0)
                tag_ll = (cGenJetTot == 0) & (bGenJetTot == 0)
                
                sampleFlavSplit = 1 * tag_cc  +  2 * tag_bb  +  3 * tag_bc  +  4 * tag_cl  +  5 * tag_bl  +  6 * tag_ll 
                selection.add('_cc',sampleFlavSplit == 1)
                selection.add('_bb',sampleFlavSplit == 2)
                selection.add('_bc',sampleFlavSplit == 3)
                selection.add('_cl',sampleFlavSplit == 4)
                selection.add('_bl',sampleFlavSplit == 5)
                selection.add('_udbsg',sampleFlavSplit == 6) # tbf I don't know why it contains b
            
            #elif dataset in ['WZTo1L1Nu2Q', 'ZZTo2L2Q', 'ZZTo2Q2Nu']: # VZ signal datasets
            elif int(sample_type) in [32,36,37]: # VZ signal datasets
                split_by_flav = True
                possible_flavSplits = ['cc','bb','ll']
                # =================================================================================
                #
                # #                       Split VZ signal by flavour, via GenPart
                #
                # ---------------------------------------------------------------------------------
                # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L2229-L2264
                gen_part = events.GenPart
                
                
                Z_decay_mothers_A = (abs(gen_part.pdgId) == 23) & (gen_part.hasFlags('isLastCopy'))
                
                Z_decays = gen_part[Z_decay_mothers_A]
                output['cutflow'][dataset]['GenPart VZ signal'] += ak.sum(Z_decay_mothers_A)
                
                n_b_from_Z = ak.sum(ak.sum(abs(Z_decays.children.pdgId) == 5, axis=-1), axis=-1)
                n_c_from_Z = ak.sum(ak.sum(abs(Z_decays.children.pdgId) == 4, axis=-1), axis=-1)
                
                
                
                VZ_cc = (n_c_from_Z >= 2)
                VZ_bb = (n_b_from_Z >= 2)
                VZ_others = (~VZ_cc) & (~VZ_bb)
                # 1, 2 and 3 identical to what was done in AnalysisTools! Do not confuse with BTV / hadron / parton flavour...
                sampleFlavSplit = 1 * VZ_cc  +  2 * VZ_bb  +  3 * VZ_others
                
                #print(sampleFlavSplit.type)
                
                selection.add('cc',sampleFlavSplit == 1)
                selection.add('bb',sampleFlavSplit == 2)
                selection.add('ll',sampleFlavSplit == 3)
            
            elif int(sample_type) in [27,28,29,30,31,33,34,35,38,39]: 
                possible_flavSplits = ['ll']
                sampleFlavSplit = sampleFlavSplit + 3
                selection.add('ll',sampleFlavSplit == 3)
                split_by_flav = True
                
            # this is how it looked in AT for comparison:
            '''
            else if( cursample->doJetFlavorSplit
                     && ( mInt("sampleIndex")==27 || mInt("sampleIndex")==28
                      || mInt("sampleIndex")==29 || mInt("sampleIndex")==30
                      || mInt("sampleIndex")==31 || mInt("sampleIndex")==33
                      || mInt("sampleIndex")==34 || mInt("sampleIndex")==35
                      || mInt("sampleIndex")==38 || mInt("sampleIndex")==39
                      )
                     ){
                        *in["sampleIndex"] = mInt("sampleIndex")*100 + 3;
            '''
        
        
        
        
        
        # =================================================================================
        #
        # #                       Reconstruct and preselect leptons
        #
        # ---------------------------------------------------------------------------------
        
        
        # Adopt from https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L3369-L3440
        # https://gitlab.cern.ch/aachen-3a/vhcc-nano/-/blob/master/VHccProducer.py#L345-389
        
        # ## Muon cuts
        ## muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        #event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1, ascending=False)]
        event_mu = events.Muon
        # looseId >= 1 or looseId seems to be the same...
        musel = ((event_mu.pt > 20) & (abs(event_mu.eta) < 2.4) & (event_mu.looseId >= 1) & (event_mu.pfRelIso04_all<0.25))
        # but 25GeV and 0.06 for 1L, xy 0.05 z 0.2, &(abs(event_mu.dxy)<0.06)&(abs(event_mu.dz)<0.2) and tightId for 1L
        event_mu = event_mu[musel]
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1, ascending=False)]
        event_mu["lep_flav"] = 13*event_mu.charge
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        # ToDo: PtCorrGeoFit
        
        # ## Electron cuts
        ## # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        #event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e = events.Electron
        elesel = ((event_e.pt > 20) & (abs(event_e.eta) < 2.5) & (event_e.mvaFall17V2Iso_WP90==1) & (event_e.pfRelIso03_all<0.25))
        # but 30GeV and WP80 for 1L
        event_e = event_e[elesel]
        # something I saw in a recent presentation, and also in AT code:
        # https://indico.desy.de/event/34473/contributions/122201/attachments/76587/98753/RTG_Meeting_01_09_22.pdf
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/VHccAnalysis/PlotWithVarial/ZllHccLowPt.py#L256-L257
        # is to require "good electrons", which means excluding some region (eta),
        # I guess it has sth to do with transition between barrel / endcap?
        event_e = event_e[(abs(event_e.eta) > 1.5660) | (abs(event_e.eta) < 1.4442)]
        event_e = event_e[ak.argsort(event_e.pt, axis=1,ascending=False)]
        event_e["lep_flav"] = 11*event_e.charge
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)
        # sorting after selecting should be faster (less computations on average)
        
        # for this channel (Zll / 2L)
        selection.add('lepsel',ak.to_numpy((nele==2)|(nmu==2)))
        
        
        
        #### build lepton pair(s)
        good_leptons = ak.with_name(
                ak.concatenate([ event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate", )
        good_leptons = good_leptons[ak.argsort(good_leptons.pt, axis=1,ascending=False)]
        leppair = ak.combinations(
                good_leptons,
                n=2,
                replacement=False,
                axis=-1,
                fields=["lep1", "lep2"],
            )
        
        ll_cand = ak.zip({
                    "lep1" : leppair.lep1,
                    "lep2" : leppair.lep2,
                    "pt": (leppair.lep1+leppair.lep2).pt,
                    "eta": (leppair.lep1+leppair.lep2).eta,
                    "phi": (leppair.lep1+leppair.lep2).phi,
                    "mass": (leppair.lep1+leppair.lep2).mass,
                    }, with_name="PtEtaPhiMLorentzVector"
                )
        # probably there needs to be a cross-check that we don't include more than we want here,
        # I know there is the option to truncate the array if more than 1 is found
        # --> clip = True
        ll_cand = ak.pad_none(ll_cand,1,axis=1)
        
        # there seem to be multiple ways to get the "one" ll_cand of interest
        # - closest to Z-mass [makes sense]
        #   I think others use this
        # - lepton-pair with highest pt [also, maybe it's even the same in the majority of the cases]
        #   https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L3369-L3440    
        
        if (ak.count(ll_cand.pt)>0):
            ll_cand  = ll_cand[ak.argsort(ll_cand.pt, axis=1,ascending=False)]
            # try the second option here
        # NOTE: Comment out to debug stuff
        ll_cand = ll_cand[:, 0]
            
        
            
        # =================================================================================
        #
        # #                       Reconstruct and preselect jets
        #
        # ---------------------------------------------------------------------------------
        
        # Apply correction:
        if isRealData:
            #print(dataset_long)
            jets =  jec(events,events.Jet,dataset_long,self._year,self._corr)
        else:
            jets =  jec(events,events.Jet,dataset,self._year,self._corr)
        #jets =  events.Jet
        
        # This was necessary for the FSR code
        #jets = jets.mask[ak.num(jets) > 2]
        
        
        
        # For EOY: recalculate CvL & CvB here, because the branch does not exist in older files
        # adapted from PostProcessor
        def deepflavcvsltag(jet):
            btagDeepFlavL = 1.-(jet.btagDeepFlavC+jet.btagDeepFlavB)
            return ak.where((jet.btagDeepFlavB >= 0.) & (jet.btagDeepFlavB < 1.) & (jet.btagDeepFlavC >= 0.) & (btagDeepFlavL >= 0.),
                            jet.btagDeepFlavC/(1.-jet.btagDeepFlavB),
                            (-1.) * ak.ones_like(jet.btagDeepFlavB))
        
        def deepflavcvsbtag(jet):
            btagDeepFlavL = 1.-(jet.btagDeepFlavC+jet.btagDeepFlavB)
            return ak.where((jet.btagDeepFlavB > 0.) & (jet.btagDeepFlavC > 0.) & (btagDeepFlavL >= 0.),
                            jet.btagDeepFlavC/(jet.btagDeepFlavC+jet.btagDeepFlavB),
                            (-1.) * ak.ones_like(jet.btagDeepFlavB))
        
        # Alternative ways:
        # - depending on the Nano version, there might already be bTagDeepFlavCvL available
        # - one could instead use DeepCSV via bTagDeepCvL
        # - not necessarily use CvL, other combination possible ( CvB | pt | BDT? )
        
        #jets["btagDeepFlavCvL"] = deepflavcvsltag(jets)
        #jets["btagDeepFlavCvB"] = deepflavcvsbtag(jets)
        jets = jets[ak.argsort(jets.btagDeepFlavCvL, axis=1, ascending=False)]

        
        # Jets are considered only if the following identification conditions hold, as mentioned in AN
        # - Here is some documentation related to puId and jetId:
        #     https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
        #     https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID
        jet_conditions = ((abs(jets.eta) < 2.4) & (jets.pt > 20) & (jets.puId > 0)) \
                     | ((jets.pt>50) & (jets.jetId>5))
        # Count how many jets exist that pass this selection
        njet = ak.sum(jet_conditions,axis=1)
        selection.add('jetsel',ak.to_numpy(njet>=2))
        
        
        # =================================================================================
        #
        # #                                 FSR recovery
        #
        # ---------------------------------------------------------------------------------
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L841-L956
        
        # FSR jets are selected with slightly different criteria
        fsr_conditions = (abs(jets.eta) < 3) & (jets.pt > 20) \
                        & ak.all(jets.metric_table(ll_cand.lep1)>0.2) & ak.all(jets.metric_table(ll_cand.lep2)>0.2)
        # Take the first two jets that pass the criteria and check the remaining ones,
        # as well as potentially others, to get FSR jets:
        pick2 = jets[ak.pad_none(ak.local_index(jets, 1)[jet_conditions], 2)[:, :2]]
        others = jets[ak.concatenate([ak.pad_none(ak.local_index(jets, 1)[(jet_conditions) & (fsr_conditions)], 2)[:, 2:], 
                                    ak.local_index(jets, 1)[(~jet_conditions) & (fsr_conditions)]
                                   ], axis=1)]
        

        def find_fsr(leading, subleading, others, threshold=0.8):
            mval1, (a1, b) = leading.metric_table(others, return_combinations=True)
            mval2, (a2, b) = subleading.metric_table(others, return_combinations=True)

            def res(mval, out):
                order = ak.argsort(mval, axis=-1)
                return out[order], mval[order]

            out1, metric1 =  res(mval1, b)
            out2, metric2 =  res(mval2, b)

            out1 = out1.mask[(metric1 <= threshold) & (metric1 < metric2)]
            out2 = out2.mask[(metric2 <= threshold) & (metric2 < metric1)]
            #out2 = out2.mask[(metric1 <= threshold) & (metric2 < metric1)]
            return out1[:, 0, ...], out2[:, 0, ...]

        
        missing = ~(ak.is_none(pick2[:, 0]) | ak.is_none(pick2[:, 1]))
        pick2 = pick2.mask[missing]
        others = others.mask[missing]

        
        leading, subleading = pick2[:, 0], pick2[:, 1]
        fsr_leading, fsr_subleading = find_fsr(leading, subleading, others, threshold=0.8)

        #print(leading.pt)
        #print((leading + fsr_leading.sum()).pt)
        
        # To explicitly check that adding FSR does indeed have an effect
        #print(ak.sum((leading + fsr_leading.sum()).pt != leading.pt))
        
        #print(leading.type)
        
        # Collect the (sub-)leading jets and their respective FSR jets in a new 4-vector
        leading_with_fsr = ak.zip({
                    "jet1" : leading,
                    "jet2" : fsr_leading.sum(),
                    "pt": (leading + fsr_leading.sum()).pt,
                    "eta": (leading + fsr_leading.sum()).eta,
                    "phi": (leading + fsr_leading.sum()).phi,
                    "mass": (leading + fsr_leading.sum()).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        subleading_with_fsr = ak.zip({
                    "jet1" : subleading,
                    "jet2" : fsr_subleading.sum(),
                    "pt": (subleading + fsr_subleading.sum()).pt,
                    "eta": (subleading + fsr_subleading.sum()).eta,
                    "phi": (subleading + fsr_subleading.sum()).phi,
                    "mass": (subleading + fsr_subleading.sum()).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        
        # (Maybe) one could calculate the angle between FSR & the "main" jet they correspond to
        # - this would be correlated with the mass of the decaying p. via the dead-cone effect,
        # - could be a discriminating variable at the event level.
        
        # =================================================================================
        #
        # #                       Build Higgs candidate w/ or w/o FSR
        #
        # ---------------------------------------------------------------------------------
        
        # Build 4-vector from leading + subleading jets, with or without FSR
        higgs_cand_no_fsr = ak.zip({
                    "jet1" : leading,
                    "jet2" : subleading,
                    "pt": (leading + subleading).pt,
                    "eta": (leading + subleading).eta,
                    "phi": (leading + subleading).phi,
                    "mass": (leading + subleading).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        higgs_cand = ak.zip({
                    "jet1" : leading_with_fsr,
                    "jet2" : subleading_with_fsr,
                    "pt": (leading_with_fsr + subleading_with_fsr).pt,
                    "eta": (leading_with_fsr + subleading_with_fsr).eta,
                    "phi": (leading_with_fsr + subleading_with_fsr).phi,
                    "mass": (leading_with_fsr + subleading_with_fsr).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        
        
        # =================================================================================
        #
        # #                       Actual event selection starts here
        #
        # ---------------------------------------------------------------------------------
        
        
        # Common global requirements in the Zll channel
        # - valid for 2LH and 2LL
        # - valid for any region, no matter if SR or CR
        
        # leppair and ll_cand have different dim, leppair contains lists,
        # ll_cand only numbers on innermost dim (because already reduced above)
        # therefore when evaluating ak.any with axis=-1,
        # ll_cand will ALWAYS be true (a.k.a. for every event), as long as one event fulfils the criterion
        # for leppair, there needs to be one per event, as expected
       # print((leppair.lep1.pt>20))
       # print((ll_cand.mass>75))
       # print((higgs_cand.mass<250))
       # print((njet>=2))
        # inside any one can then only place stuff that has one more dim
        
        # related to individual leptons
        req_global = ak.any((leppair.lep1.pt>20) & (leppair.lep2.pt>20) \
                              # opposite charge
                        & ((leppair.lep1.charge+leppair.lep2.charge)==0) \
                        , axis=-1
            )
        # cands and global stuff
        # note: V_pt > 60 as in AT, AN: 50 (don't confuse)
        req_global = req_global \
                   & (ll_cand.pt>60) \
                   & (njet>=2) \
                   & (higgs_cand.mass<250)
        
        
        selection.add('global_selection',ak.to_numpy(req_global))
        
        
        mask2e = req_global & (nele == 2)
        mask2mu = req_global & (nmu == 2)
        
        #mask2lep = [ak.any(tup) for tup in zip(maskemu, mask2mu, mask2e)]
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        
        good_leptons = ak.mask(good_leptons,mask2lep)
       
        
        #output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        
        selection.add('ee',ak.to_numpy(nele == 2))
        selection.add('mumu',ak.to_numpy(nmu == 2))
        
        
        #print(higgs_cand.type)
        #print(ll_cand.type)
        
        # global already contains Vpt>60 as the lower bound
        # global also has higgs_cand.mass<250
        req_sr_Zll = (ll_cand.mass > 75) & (ll_cand.mass < 105) \
                            & (higgs_cand.delta_phi(ll_cand)>2.5) \
                            & (higgs_cand.mass>=50) & (higgs_cand.mass<=200) \
                            & (leading.btagDeepFlavCvL>0.225) & (leading.btagDeepFlavCvB>0.4)
        # flip H mass, otherwise same
        req_cr_Zcc = (ll_cand.mass > 85) & (ll_cand.mass < 97) \
                            & (higgs_cand.delta_phi(ll_cand)>2.5) \
                            & ~((higgs_cand.mass>=50) & (higgs_cand.mass<=200)) \
                            & (leading.btagDeepFlavCvL>0.225) & (leading.btagDeepFlavCvB>0.4)
        # Note: m_ll requirement not in AN, but in AT
        req_cr_Z_LF = (ll_cand.mass > 75) & (ll_cand.mass < 105) \
                            & (higgs_cand.delta_phi(ll_cand)>2.5) \
                            & (higgs_cand.mass>=50) & (higgs_cand.mass<=200) \
                            & (leading.btagDeepFlavCvL<0.225) & (leading.btagDeepFlavCvB>0.4)
        
        req_cr_Z_HF = (ll_cand.mass > 85) & (ll_cand.mass < 97) \
                            & (higgs_cand.delta_phi(ll_cand)>2.5) \
                            & (higgs_cand.mass>=50) & (higgs_cand.mass<=200) \
                            & (leading.btagDeepFlavCvL>0.225) & (leading.btagDeepFlavCvB<0.4)
        
        req_cr_t_tbar = ~((ll_cand.mass>0) & (ll_cand.mass<10)) & ~((ll_cand.mass>75) & (ll_cand.mass<120)) \
                            & (higgs_cand.mass>=50) & (higgs_cand.mass<=200) \
                            & (leading.btagDeepFlavCvL>0.225) & (leading.btagDeepFlavCvB<0.4)
        
        req_sr_Zll_vpt_low  = req_global & req_sr_Zll & (ll_cand.pt<150)
       # print(ll_cand.pt<150)
       # print(ak.any(ll_cand.pt<150, axis=-1)
       # print(req_sr_Zll_vpt_low)
        req_sr_Zll_vpt_high = req_global & req_sr_Zll & (ll_cand.pt>150)
       # print(ll_cand.pt>150)
       # print(req_sr_Zll_vpt_high)
       # print(len(req_sr_Zll_vpt_low))
       # print(len(req_sr_Zll_vpt_low == req_sr_Zll_vpt_high))
       # print(np.sum(ak.to_numpy(req_sr_Zll_vpt_low)))
       # print(np.sum(ak.to_numpy(req_sr_Zll_vpt_low == req_sr_Zll_vpt_high)))
        
        req_cr_Zcc_vpt_low  = req_global & req_cr_Zcc & (ll_cand.pt<150)
       # print(req_sr_Zll_vpt_low)
        req_cr_Zcc_vpt_high = req_global & req_cr_Zcc & (ll_cand.pt>150)
       # print(req_sr_Zll_vpt_high)
       # print(np.sum(ak.to_numpy(req_sr_Zll_vpt_low & req_sr_Zll_vpt_high)))
        
        req_cr_Z_LF_vpt_low  = req_global & req_cr_Z_LF & (ll_cand.pt<150)
        req_cr_Z_LF_vpt_high = req_global & req_cr_Z_LF & (ll_cand.pt>150)
        
        req_cr_Z_HF_vpt_low  = req_global & req_cr_Z_HF & (ll_cand.pt<150)
        req_cr_Z_HF_vpt_high = req_global & req_cr_Z_HF & (ll_cand.pt>150)
        
        req_cr_t_tbar_vpt_low  = req_global & req_cr_t_tbar & (ll_cand.pt<150)
        req_cr_t_tbar_vpt_high = req_global & req_cr_t_tbar & (ll_cand.pt>150)
        
        
        #prob not necessary
        #selection.add('SR',ak.to_numpy(req_sr_Zll))
        
        selection.add('SR_2LL',ak.to_numpy(req_sr_Zll_vpt_low))
        selection.add('SR_2LH',ak.to_numpy(req_sr_Zll_vpt_high))
        selection.add('CR_Zcc_2LL',ak.to_numpy(req_cr_Zcc_vpt_low))
        selection.add('CR_Zcc_2LH',ak.to_numpy(req_cr_Zcc_vpt_high))
        selection.add('CR_Z_LF_2LL',ak.to_numpy(req_cr_Z_LF_vpt_low))
        selection.add('CR_Z_LF_2LH',ak.to_numpy(req_cr_Z_LF_vpt_high))
        selection.add('CR_Z_HF_2LL',ak.to_numpy(req_cr_Z_HF_vpt_low))
        selection.add('CR_Z_HF_2LH',ak.to_numpy(req_cr_Z_HF_vpt_high))
        selection.add('CR_t_tbar_2LL',ak.to_numpy(req_cr_t_tbar_vpt_low))
        selection.add('CR_t_tbar_2LH',ak.to_numpy(req_cr_t_tbar_vpt_high))
        
        
        
        
        
        # =================================================================================
        #
        # #                    Calculate and store weights & factors
        #
        # ---------------------------------------------------------------------------------
        
        # there is also nProcEvents, which might be related to nEvents by some factor
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/HelperClasses/SampleContainer.cc
        # there are some more calculations related to weights, e.g.
        # https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/HelperClasses/SampleContainer.cc#L115-L154
        
        # ToDo:
        # [ ] LHEScaleWeight ??
        # [ ] intWeight - is this only relevant when running over the post-processed samples, or already on top of Nano+AK15?
        # [x] genWeight
        # [ ] PrefireWeight - (for 2016+2017) see also:
        #     https://github.com/mastrolorenzo/AnalysisTools-1/blob/master/plugins/VHccAnalysis.cc#L2099-L2113
        # [ ] weight_PU
        # [ ] weight_ptEWK
        # [(x)] Lep_SF - but I'm not sure about EOY / UL compatibility
        # [ ] recoWReWeight
        # [ ] WJetNLOWeight
        # [ ] cTagWeight - later, also including up/down syst
        # [ ] weight_mettrigSF
        # [ ] weight_puid - not the same as _PU
        # [ ] weight_subptEWKnnlo - find out what "SubGen" is
        #
        # [ ] LOtoNLOWeightBjetSplitEtabb
        # [ ] WPtCorrFactor
        # [ ] ZPtCorrFactor
        
        
        
        
        # running over more than just the Double[] datasets, but still requiring the same trigger
        # not sure if correct
        if 'DoubleEG' in dataset or 'Electron' in dataset:
            output['cutflow'][dataset]['trigger'] += ak.sum(trigger_ee)
        elif 'Muon' in dataset :
            output['cutflow'][dataset]['trigger'] += ak.sum(trigger_mm)
            
            
        # Successively add another cut w.r.t. previous line, looks a bit like N-1 histograms
        output['cutflow'][dataset]['jet selection'] += ak.sum(njet>=2)
        output['cutflow'][dataset]['global selection'] += ak.sum(req_global)
        output['cutflow'][dataset]['signal region'] += ak.sum(req_global & req_sr_Zll)
        output['cutflow'][dataset]['signal region & ee or mumu'] += ak.sum(req_global & req_sr_Zll & ( ((nele == 2) & trigger_ee) | ((nmu == 2) & trigger_mm)))
        output['cutflow'][dataset]['signal ee'] += ak.sum(req_global & req_sr_Zll & (nele == 2) & trigger_ee)
        output['cutflow'][dataset]['signal mumu'] += ak.sum(req_global & req_sr_Zll & (nmu == 2) & trigger_mm)
        

        lepflav = ['ee','mumu']
        reg = ['SR_2LL','SR_2LH',
               'CR_Zcc_2LL','CR_Zcc_2LH',
               'CR_Z_LF_2LL','CR_Z_LF_2LH',
               'CR_Z_HF_2LL','CR_Z_HF_2LH',
               'CR_t_tbar_2LL','CR_t_tbar_2LH']
        
        #print(possible_flavSplits)
        list_weights = []
        lists_of_vars = {}
        names = ['wei', 'Higgs_mass', 'Higgs_pt', 'Z_mass', 'Z_pt', 'jjVptratio', 'CvsL_max',
                 'CvsL_min', 'CvsB_max', 'CvsB_min', 'pt_lead', 'pt_sublead',
                 'del_phi_jjV', 'del_R_jj', 'del_phi_jj', 'del_eta_jj', 'del_phi_ll', 'del_eta_ll',
                 'del_phi_l2_subleading', 'del_phi_l2_leading']
        modes = ['low_ee', 'high_ee', 'low_mumu', 'high_mumu']
        for name in names:
            for mode in modes:
                lists_of_vars[f'{name}_{mode}'] = []
        '''
        lists_of_vars = {'wei': [],
                 'Higgs_mass':  [],
                 'Higgs_pt': [],
                 'Z_pt': [],
                 'jjVptratio': [],
                 'CvsL_max': [],
                 'CvsL_min': [], 
                 'CvsB_max': [],
                 'CvsB_min': [],
                 'pt_lead': [],
                 'pt_sublead': [],
                 'del_phi_jjV': [],
                 'del_R_jj': [], 
                 'del_eta_jj': [],
                 'del_phi_ll': [], 
                 'del_eta_ll': [],
                 'del_phi_l2_subleading': [],
                 'del_phi_l2_leading': []
                 }
        '''
        #### write into histograms (i.e. write output)
        for histname, h in output.items():
            for s in possible_flavSplits:
                dataset_renamed = dataset if s == 'already_split_sample' else dataset + s
                for ch in lepflav:
                    for r in reg:
                        cut = selection.all('lepsel',
                                            'jetsel',
                                            'global_selection',
                                            'metfilter',
                                            'lumi',
                                            r,
                                            ch,
                                            s,
                                            'trigger_%s'%(ch))
                        llcut = ll_cand[cut]
                        # this next line is necessary if running with multiple possible ll candidates
                        #llcut = llcut[:,0]

                        lep1cut = llcut.lep1
                        lep2cut = llcut.lep2
                        #print(self._version)
                        if not isRealData and not self._debug:
                            #print('not data, not test')
                            if ch == 'ee':
                                lepsf = eleSFs(lep1cut, self._year, self._corr) * eleSFs(lep2cut, self._year, self._corr)
                            elif ch == 'mumu':
                                lepsf = muSFs(lep1cut, self._year, self._corr) * muSFs(lep2cut, self._year, self._corr)
                            '''
                            # This would be emu channel, which does not exist in the VHcc Zll case
                            else:
                                lepsf = np.where(lep1cut.lep_flav == 11,
                                               eleSFs(lep1cut, self._year, self._corr) * muSFs(lep2cut, self._year, self._corr),
                                               1.) \
                                      * np.where(lep1cut.lep_flav == 13,
                                               eleSFs(lep2cut, self._year, self._corr) * muSFs(lep1cut, self._year, self._corr),
                                               1.)
                           '''
                        else : 
                            #lepsf = weights.weight()[cut]
                            # AS: if I understand correctly, this only works because in case of data, weights are identically 1 for every entry
                            # otherwise this would double count the weights in a later step (where lepsf gets multiplied by the weights!)
                            lepsf = ak.full_like(weights.weight()[cut], 1)
                        #print(lepsf)
                        # print(weights.weight()[cut]*lepsf)
                        # print(lepsf)
                        '''
                        if self._export_array and not isRealData:
                            if ch == 'ee' and r == 'SR_2LL' and s == '_cc':
                                            eell_cand = ak.zip({
                                                           "Higgs_mass" : higgs_cand['mass'][cut] * lepsf,
                                                           #"jet2" : subleading_with_fsr,
                                                           #"pt": (leading_with_fsr + subleading_with_fsr).pt,
                                                           #"eta": (leading_with_fsr + subleading_with_fsr).eta,
                                                           #"phi": (leading_with_fsr + subleading_with_fsr).phi,
                                                           #"mass": (leading_with_fsr + subleading_with_fsr).mass,
                                                            })
                                            print(eell_cand)
                                            '''
                        if 'leading_jetflav_' in histname and 'sub' not in histname:
                            #print(dir(leading))
                            #print(h.axes)
                            names = [ax.name for ax in h.axes]
                            fields = {l: normalize(leading[histname.replace('leading_jetflav_','')],
                                                   cut) for l in names if l in dir(leading)}
                            #print(fields)
                            #sys.exit()
                            if isRealData:
                                flavor = ak.zeros_like(normalize(leading['pt'],cut))
                            else:
                                flavor = normalize(leading.hadronFlavour,cut)
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   flav = flavor,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf)  
                        elif 'subleading_jetflav_' in histname:
                            #print(dir(subleading))
                            names = [ax.name for ax in h.axes]
                            fields = {l: normalize(subleading[histname.replace('subleading_jetflav_','')],
                                                   cut) for l in names if l in dir(subleading)}
                            if isRealData:
                                flavor = ak.zeros_like(normalize(subleading['pt'],cut))
                            else:
                                flavor = normalize(subleading.hadronFlavour,cut)
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   flav = flavor,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf)  
                        elif 'lep1_' in histname:
                            names = [ax.name for ax in h.axes]
                            fields = {l: ak.fill_none(flatten(lep1cut[histname.replace('lep1_','')]),
                                                      np.nan) for l in names if l in dir(lep1cut)}
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf)
                        elif 'lep2_' in histname:
                            names = [ax.name for ax in h.axes]
                            fields = {l: ak.fill_none(flatten(lep2cut[histname.replace('lep2_','')]),
                                                      np.nan) for l in names if l in dir(lep2cut)}
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf)
                        #elif 'MET_' in histname:
                        #    fields = {l: normalize(events.MET[histname.replace('MET_','')],
                        #                           cut) for l in names if l in dir(events.MET)}
                        #    h.fill(
                        #           datasetSplit = dataset_renamed,
                        #           lepflav = ch,
                        #           region = r,
                        #           **fields,
                        #           weight = weights.weight()[cut] * lepsf) 
                        elif 'll_' in histname:
                            names = [ax.name for ax in h.axes]
                            fields = {l: ak.fill_none(flatten(llcut[histname.replace('ll_','')]),
                                                      np.nan) for l in names if l in dir(llcut)}
                            #print(max(llcut['pt']))
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf) 
                        elif 'jj_' in histname:
                            names = [ax.name for ax in h.axes]
                            fields = {l: normalize(higgs_cand[histname.replace('jj_','')],
                                                   cut) for l in names if l in dir(higgs_cand)}
                            h.fill(
                                   datasetSplit = dataset_renamed,
                                   lepflav = ch,
                                   region = r,
                                   **fields,
                                   weight = weights.weight()[cut] * lepsf) 
                        else:
                            output['nj'].fill(
                                              datasetSplit = dataset_renamed,
                                              lepflav = ch,
                                              region = r,
                                              nj = normalize(ak.num(jet_conditions),cut),
                                              weight = weights.weight()[cut]*lepsf)
                            # check?
                            output['nAddJets302p5_puid'].fill(
                                                              datasetSplit = dataset_renamed,
                                                              lepflav = ch,
                                                              region = r,
                                                              nAddJets302p5_puid = normalize(ak.where((ak.num(jet_conditions) > 2),
                                                                                                    (ak.num(jet_conditions)-2),
                                                                                                    (ak.zeros_like(ak.num(jet_conditions)))
                                                                                                    ),
                                                                                             cut),
                                                              weight = weights.weight()[cut]*lepsf)
                            # check?
                            output['nAddJetsFSRsub302p5_puid'].fill(
                                                                    datasetSplit = dataset_renamed,
                                                                    lepflav = ch,
                                                                    region = r,
                                                                    nAddJetsFSRsub302p5_puid = normalize(ak.where((ak.where((ak.num(jet_conditions) > 2),
                                                                                                                            (ak.num(jet_conditions)-2),
                                                                                                                            (ak.zeros_like(ak.num(jet_conditions)))
                                                                                                                            )
                                                                                                                     -ak.num((~jet_conditions) & (fsr_conditions))) > 0, 
                                                                                                                 (ak.where((ak.num(jet_conditions) > 2),
                                                                                                                            (ak.num(jet_conditions)-2),
                                                                                                                            (ak.zeros_like(ak.num(jet_conditions)))
                                                                                                                            )
                                                                                                                     -ak.num((~jet_conditions) & (fsr_conditions))),
                                                                                                                 (ak.zeros_like(ak.num(jet_conditions)))),
                                                                                                         cut),
                                                                    weight = weights.weight()[cut]*lepsf)
                            #if not isRealData:
                            #    output['weight_full'].fill(
                            #                               datasetSplit = dataset_renamed,
                            #                               lepflav = ch,
                            #                               region = r,
                            #                               weight_full = weights.weight()[cut]*lepsf)
                            #    output['genweight'].fill(
                            #                             datasetSplit = dataset_renamed,
                            #                             lepflav = ch,
                            #                             region = r,
                            #                             genWeight = events.genWeight[cut])
                            #    output['sign_genweight'].fill(
                            #                                  datasetSplit = dataset_renamed,
                            #                                  lepflav = ch,
                            #                                  region = r,
                            #                                  genWeight_by_abs = (events.genWeight/abs(events.genWeight))[cut])
                            output['jjVPtRatio'].fill(
                                                      datasetSplit = dataset_renamed,
                                                      lepflav = ch,
                                                      region = r,
                                                      jjVPtRatio = (normalize(higgs_cand['pt'],
                                                                              cut) / ak.fill_none(flatten(llcut['pt']),
                                                                                                  np.nan)),
                                                      weight = weights.weight()[cut] * lepsf)
                            if self._export_array and not isRealData:
                                import pandas as pd
                                #output['array'][dataset]['weight'] += processor.column_accumulator(
                                #                                        ak.to_numpy(weights.weight()[cut] * lepsf)
                                #                                      )
                                
                                list_weights.append(ak.to_numpy(weights.weight()[cut] * lepsf))
                                
                                roi = ['SR_2LL','SR_2LH']
                                lepflav_chosen = ['ee','mumu']
                                names_dict = {'wei': weights.weight()[cut] * lepsf,
                                              'Higgs_mass':  higgs_cand['mass'][cut] * lepsf,
                                              'Higgs_pt': higgs_cand['pt'][cut] * lepsf,
                                              'Z_mass': ll_cand['mass'][cut] * lepsf,
                                              'Z_pt': ll_cand['pt'][cut] * lepsf,
                                              'jjVptratio': (higgs_cand['pt'][cut] * lepsf)/ (ll_cand['pt'][cut] * lepsf),
                                              'CvsL_max': leading_with_fsr['jet1']['btagDeepFlavCvL'][cut] * lepsf,
                                              'CvsL_min': subleading_with_fsr['jet1']['btagDeepFlavCvL'][cut] * lepsf, 
                                              'CvsB_max': leading_with_fsr['jet1']['btagDeepFlavCvB'][cut] * lepsf,
                                              'CvsB_min': subleading_with_fsr['jet1']['btagDeepFlavCvB'][cut] * lepsf,
                                              'pt_lead': leading_with_fsr['jet1']['pt'][cut] * lepsf,
                                              'pt_sublead': subleading_with_fsr['jet1']['pt'][cut] * lepsf,
                                              'del_phi_jjV': np.abs((higgs_cand[cut] * lepsf).delta_phi((ll_cand[cut] * lepsf))),
                                              'del_R_jj': np.abs((higgs_cand['jet1'][cut] * lepsf).delta_r((higgs_cand['jet2'][cut] * lepsf))), 
                                              'del_eta_jj': np.abs((higgs_cand['jet1']['eta'][cut] * lepsf) - ((higgs_cand['jet2']['eta'][cut] * lepsf))),
                                              'del_phi_jj': np.abs((higgs_cand['jet1'][cut] * lepsf).delta_phi((higgs_cand['jet2'][cut] * lepsf))),
                                              'del_phi_ll': np.abs((ll_cand['lep1'][cut] * lepsf).delta_phi((ll_cand['lep2'][cut] * lepsf))), 
                                              'del_eta_ll': np.abs((ll_cand['lep1']['eta'][cut] * lepsf) - ((ll_cand['lep2']['eta'][cut] * lepsf))),
                                              'del_phi_l2_subleading': np.abs((ll_cand['lep2'][cut] * lepsf).delta_phi((higgs_cand['jet1'][cut] * lepsf))),
                                              'del_phi_l2_leading': np.abs((ll_cand['lep2'][cut] * lepsf).delta_phi((higgs_cand['jet2'][cut] * lepsf)))
                                               }
                                if ch in lepflav_chosen and r in roi:
                                    if ch == 'ee':
                                        if r == 'SR_2LL':
                                            for var_name, var_value in names_dict.items():
                                                lists_of_vars[f'{var_name}_low_ee'].append(ak.to_numpy(var_value))
                                                #output['array'][dataset][f'{var_name}_low_ee'] += processor.column_accumulator(
                                                #                        ak.to_numpy(var_value)
                                                #                     )
                                        elif r == 'SR_2LH':
                                            for var_name, var_value in names_dict.items():
                                                lists_of_vars[f'{var_name}_high_ee'].append(ak.to_numpy(var_value))
                                                #output['array'][dataset][f'{var_name}_high_ee'] += processor.column_accumulator(
                                                #                        ak.to_numpy(var_value)
                                                #                      )
                                    elif ch == 'mumu':
                                        if r == 'SR_2LL':
                                            for var_name, var_value in names_dict.items():
                                                lists_of_vars[f'{var_name}_low_mumu'].append(ak.to_numpy(var_value))
                                                #output['array'][dataset][f'{var_name}_low_mumu'] += processor.column_accumulator(
                                                #                        ak.to_numpy(var_value)
                                                #                      )
                                        elif r == 'SR_2LH':
                                            for var_name, var_value in names_dict.items():
                                                lists_of_vars[f'{var_name}_high_mumu'].append(ak.to_numpy(var_value))
                                                #output['array'][dataset][f'{var_name}_high_mumu'] += processor.column_accumulator(
                                                #                        ak.to_numpy(var_value)
                                                #                      )
                                            
                                
                                
                                               ### + regression, kinfit ???
        list_weights = np.array([item for sublist in list_weights for item in sublist])
        print(list_weights)
        #print(lists_of_vars)
        for v_name in lists_of_vars.keys():
            lists_of_vars[v_name] = np.array([item for sublist in lists_of_vars[v_name] for item in sublist])
        print(lists_of_vars)
        if 'ZH' in dataset:
            ttyp = 'signal_03'
        else:
            ttyp = 'back_03'
        folder_save = f'condor_{ttyp}'
        if not os.path.exists(f"./{folder_save}"):
            os.mkdir(f"./{folder_save}")
        if not os.path.exists(f"./{folder_save}/{dataset}"):
            os.mkdir(f"./{folder_save}/{dataset}")
        if not os.path.exists(f"./{folder_save}/{dataset}/{filename}"):
            os.mkdir(f"./{folder_save}/{dataset}/{filename}")
        try:
            df_weights = pd.read_csv(f'{folder_save}/{dataset}/{filename}/test_save_weights_full.csv')
        except FileNotFoundError:
            df_weights = pd.DataFrame([], columns = ['weights'])
        df_wei = pd.DataFrame([], columns = ['weights'])
        df_wei['weights'] = list_weights
        df_weights_full = pd.concat([df_weights, df_wei], ignore_index = True)
        df_wei.to_csv(f'{folder_save}/{dataset}/{filename}/test_save_weights.csv', sep=',', encoding='utf-8', index=False)
        df_weights_full.to_csv(f'{folder_save}/{dataset}/{filename}/test_save_weights_full.csv', sep=',', encoding='utf-8', index=False)
        try:
            df_else_everything = pd.read_csv(f'{folder_save}/{dataset}/{filename}/test_else_save_no_weights_full.csv')
        except FileNotFoundError:
            df_else_everything = pd.DataFrame([], columns = [v_name for v_name in lists_of_vars.keys()])
        df_else = pd.DataFrame([], columns = [v_name for v_name in lists_of_vars.keys()])
        #print(df_else)
        for var in lists_of_vars.keys():
            df_else[var] = pd.Series(lists_of_vars[var])
        df_else_full = pd.concat([df_else_everything, df_else], ignore_index = True)
        
        df_else.to_csv(f'{folder_save}/{dataset}/{filename}/test_else_save_no_weights.csv', sep=',', encoding='utf-8', index=False)
        df_else_full.to_csv(f'{folder_save}/{dataset}/{filename}/test_else_save_no_weights_full.csv', sep=',', encoding='utf-8', index=False)                                              
                    
        return {dataset: output}

    def postprocess(self, accumulator):
        #print(accumulator)
        return accumulator
