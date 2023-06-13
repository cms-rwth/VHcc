#!/usr/bin/env python3

from os import listdir, makedirs, path, system, getpid
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
import pickle as pkl
import hist as Hist
import awkward as ak
from matplotlib import pyplot as plt
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights, PackedSelection
from functools import partial
import psutil
from BTVNanoCommissioning.utils.correction import (
    met_filters, load_lumi, load_SF, muSFs, eleSFs, puwei,
)
processor.NanoAODSchema.warn_missing_crossrefs = False

class NanoProcessor(processor.ProcessorABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._campaign = self.cfg.dataset["campaign"]
        self._debug_level =  self.cfg.user["debug_level"]
        self._met_filters = met_filters['2017_UL']
        #self._pu = load_pu(self._campaign, self.cfg.weights_config["PU"])
        self.isCorr = True
        
        self.systematics = self.cfg.systematic        
        self.SF_map = load_SF(self._campaign, self.cfg.weights_config)
        print('SF_map:', self.SF_map)

        print("Year and campaign:", self._year, self._campaign)

        lepflav_axis = Hist.axis.StrCategory(
            ['mu', 'el', 'emu'], name="lepflav", label="channel"
        )

        jetflav_axis = Hist.axis.StrCategory(
            ['jj','cj','bj','cc','cb','bb','oo'], name="jetflav", label="Jet flavors"
        )
        mBin_axis = Hist.axis.Variable([0,60,120,2000], name="dijet_mBin", label="dijet_mBin")

        pt_axis   = Hist.axis.Regular(50, 0, 150, name="pt", label=r"$p_{T}$ [GeV]")
        #eta_axis  = Hist.axis.Regular(25, -2.6, 2.6, name="eta", label=r"$\eta$")
        #phi_axis  = Hist.axis.Regular(30, -3.2, 3.2, name="phi", label=r"$\phi$")
        mass_axis = Hist.axis.Regular(50, 0, 300, name="mass", label=r"$m$ [GeV]")
        #mt_axis   = Hist.axis.Regular(30, 0, 300, name="mt", label=r"$m_{T}$ [GeV]")
        dr_axis   = Hist.axis.Regular(50, 0, 5, name="dr", label=r"$\Delta$R")
        npv_axis  = Hist.axis.Integer(0,100, name="npv", label="N primary vertices")

        single_axis = {
            "LHE_Vpt": Hist.axis.Regular(
                100, 0, 400, name="LHE_Vpt", label="LHE V PT [GeV]"
            ),
            "LHE_HT": Hist.axis.Regular(
                100, 0, 400, name="LHE_HT", label="LHE HT [GeV]"
            ),
            "wei": Hist.axis.Regular(100, -1000, 10000, name="wei", label="weight"),
            "wei_sign": Hist.axis.Regular(50, -2, 2, name="wei", label="weight"),
            "nlep": Hist.axis.Regular(12, 0, 6, name="nlep", label="Number of Leptons"),
            "ndilep": Hist.axis.Regular(12, 0, 6, name="ndilep", label="Number of di-lepton pairs"),
        }
        multi_axis = {
            "dilep_m": Hist.axis.Regular(40, 70, 110, name="dilep_m", label="dilep_m"),
            "dilep_pt": Hist.axis.Regular(100, 0, 400, name="dilep_pt", label="dilep_pt"),
            "dilep_dr": Hist.axis.Regular(60, 0, 5, name="dilep_dr", label="dilep_dr"),
            "njet25": Hist.axis.Regular(12, 0, 6, name="njet25", label="njet25"),

            "dijet_m": Hist.axis.Regular(50, 0, 800, name="dijet_m", label="dijet_m"),
            "dijet_pt": Hist.axis.Regular(100, 0, 400, name="dijet_pt", label="dijet_pt"),
            "dijet_dr": Hist.axis.Regular(60, 0, 5, name="dijet_dr", label="dijet_dr"),
            #'dijet_dr_neg': Hist.axis.Regular(50, 0, 5,    name="dijet_dr", label="dijet_dr")
        }

        histDict1 = {
            observable: Hist.Hist(var_axis, lepflav_axis, jetflav_axis, name="Counts", storage="Weight") if 'dijet' in observable
            else Hist.Hist(var_axis, lepflav_axis, name="Counts", storage="Weight")
            for observable, var_axis in multi_axis.items()
        }
        histDict2 = {
            observable: Hist.Hist(var_axis, name="Counts", storage="Weight")
            for observable, var_axis in single_axis.items()
        }

        histDict3 = { "lep1_pt": Hist.Hist(pt_axis, lepflav_axis, jetflav_axis, name="Lep1_Pt", storage="Weight"),
                      "lep2_pt": Hist.Hist(pt_axis, lepflav_axis, jetflav_axis, name="Lep2_Pt", storage="Weight"),
                      "jet1_pt": Hist.Hist(pt_axis, lepflav_axis, jetflav_axis, name="Jet1_Pt", storage="Weight"),
                      "jet2_pt": Hist.Hist(pt_axis, lepflav_axis, jetflav_axis, name="Jet2_Pt", storage="Weight"),
                      "jet1_dRlep": Hist.Hist(dr_axis, lepflav_axis, jetflav_axis, name="dR_Jet1_Lep", storage="Weight"),
                      "jet2_dRlep": Hist.Hist(dr_axis, lepflav_axis, jetflav_axis, name="dR_Jet2_Lep", storage="Weight"),
                      "npv0": Hist.Hist(npv_axis, lepflav_axis,  name="npv0", storage="Weight"),
                      "npv1": Hist.Hist(npv_axis, lepflav_axis,  name="npv1", storage="Weight"),
                  }

        histDict2D = {
            "dijet_dr_mjj": Hist.Hist(multi_axis['dijet_dr'], mBin_axis, lepflav_axis, jetflav_axis, name="Counts", storage="Weight")
        }
        self._accumulator = processor.dict_accumulator(histDict1|histDict2|histDict3|histDict2D)

        self._accumulator["cutflow"] = processor.defaultdict_accumulator(
            partial(processor.defaultdict_accumulator, int)
        )
        self._accumulator["sumw"] = 0

        print(
            "\t Init : ", psutil.Process(getpid()).memory_info().rss / 1024**2, "MB"
        )

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
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8'
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
        self._lumiMasks = {
            '2016': LumiMask('src/VHcc/data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
            '2017': LumiMask('src/VHcc/data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
            '2018': LumiMask('src/VHcc/data/Lumimask/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator
        # print(output)

        dataset = events.metadata["dataset"]

        nEvents = len(events)
        output["cutflow"][dataset]["all_events"] += nEvents
        output["cutflow"][dataset]["number_of_chunks"] += 1

        isRealData = not hasattr(events, "genWeight")
        if isRealData:
            events["genWeight"] = np.ones(nEvents)

        # print(dataset)

        weights = Weights(nEvents, storeIndividual=True)
        weight_nosel = np.ones(nEvents)
        if isRealData:
            weights.add('genWeight', np.ones(nEvents))
        else:
            weights.add('genWeight', np.sign(events.genWeight))
            #if dataset in ["DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]:
            #    weights.add('genWeight', events.genWeight)
            #else:
            #    weights.add('genWeight', np.sign(events.genWeight))


        if isRealData:
            output['sumw'] += nEvents
        else:
            output["sumw"] += np.sum(weights.weight())

        if self._debug_level > 0:
            print("\n", dataset, "wei:", weights.weight())

        selection = PackedSelection()

        req_lumi = np.ones(nEvents, dtype='bool')
        if isRealData:
            req_lumi = self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumi',ak.to_numpy(req_lumi))
        del req_lumi

        # -----------------
        # Trigger selection
        # =================
        trigger_ee = np.zeros(nEvents, dtype='bool')
        trigger_mm = np.zeros(nEvents, dtype='bool')

        for t in self._mumu_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]

        for t in self._ee_hlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]

        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mm', ak.to_numpy(trigger_mm))
        #del trigger_ee
        #del trigger_mm

        # ----------
        # MET filetrs
        #===========
        metfilter = np.ones(len(events), dtype="bool")
        for flag in self._met_filters["data" if isRealData else "mc"]:
            metfilter &= np.array(events.Flag[flag])
        selection.add("metfilter", metfilter)
        del metfilter


        if isRealData:
            LHE_Vpt = np.ones(nEvents)
            LHE_HT  = np.ones(nEvents)
            #LHE_Vpt = np.random.lognormal(5, 100, nEvents)
            #LHE_HT  = np.random.lognormal(5, 100, nEvents)
        else:
            LHE_Vpt = events.LHE["Vpt"]
            LHE_HT  = events.LHE["HT"]

        output["LHE_Vpt"].fill(LHE_Vpt=LHE_Vpt, weight=weights.weight())
        output["LHE_HT"].fill(LHE_HT=LHE_HT, weight=weights.weight())

        output["wei"].fill(wei=weights.weight(), weight=weights.weight())
        output["wei_sign"].fill(
            wei=weights.weight() / np.abs(weights.weight()), weight=np.abs(weights.weight())
        )

        # --------------
        # Lepton selection
        # ==============

        muons = events.Muon
        musel = ((muons.pt > 20) & (abs(muons.eta) < 2.4) & (muons.tightId >= 1) & (muons.pfRelIso04_all<0.25) &
                 (abs(muons.dxy) < 0.06) & (abs(muons.dz)<0.2) )
        # but 25GeV and 0.06 for 1L, xy 0.05 z 0.2, &(abs(muons.dxy)<0.06)&(abs(muons.dz)<0.2) and tightId for 1L
        muons = muons[musel]
        #muons = muons[ak.argsort(muons.pt, axis=1, ascending=False)]
        muons["lep_flav"] = 13
        #muons = ak.pad_none(muons, 2, axis=1)
        nmu   = ak.sum(musel, axis=1)


        electrons = events.Electron
        elesel = ((electrons.pt > 20) & (abs(electrons.eta) < 2.5) & ((abs(electrons.eta) > 1.5660) | (abs(electrons.eta) < 1.4442)) &
                  (electrons.mvaFall17V2noIso_WPL==1) & (abs(electrons.dxy) < 0.05) & (abs(electrons.dz) < 0.1) & 
                  (electrons.convVeto==1) & (electrons.lostHits<2) & 
                  ( ( (electrons.sieie<0.011) & (abs(electrons.eta) < 1.4442) ) | ( (electrons.sieie<0.030) & (abs(electrons.eta) > 1.5660) ) )
                  & (electrons.hoe < 0.10) & (electrons.eInvMinusPInv>-0.04) & (electrons.tightCharge>=1) & (electrons.mvaTTH>0.25)
        )
        electrons = electrons[elesel]
        #electrons = electrons[ak.argsort(electrons.pt, axis=1, ascending=False)]
        electrons["lep_flav"] = 11
        #electrons = ak.pad_none(electrons, 2, axis=1)
        nel       = ak.sum(elesel, axis=1)

        output["nlep"].fill(nlep=(nmu+nel), weight=weights.weight())

        selection.add('twoLep', ak.to_numpy((nel>=2)|(nmu>=2)))

        # ---------------------
        # Build lepton pairs: the dileptons
        # =====================

        leptons = ak.with_name(ak.concatenate([ muons, electrons], axis=1), "PtEtaPhiMCandidate")
        leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

        dileptons = ak.combinations(leptons, 2, fields=["lep1", "lep2"])

        pt_cut = (dileptons["lep1"].pt > 30) | (dileptons["lep2"].pt > 20)
        #Zmass_cut = np.abs( (dileptons["lep1"] + dileptons["lep2"]).mass - 91.19) < 15
        Zmass_cut = ((dileptons["lep1"] + dileptons["lep2"]).mass > 60) & ((dileptons["lep1"] + dileptons["lep2"]).mass < 120)
        Vpt_cut = (dileptons["lep1"] + dileptons["lep2"]).pt > self.cfg.user['cuts']['vpt']
        charge_cut = (dileptons["lep1"].charge*dileptons["lep2"].charge < 0)
        #charge_cut = True

        dileptonMask = pt_cut & Zmass_cut & Vpt_cut & charge_cut
        good_dileptons = dileptons[dileptonMask]

        #print('good_dileptons', ak.num(good_dileptons), good_dileptons)
        #two_lep = ak.num(good_dileptons) >= 1
        #print('two_lep', two_lep)
        selection.add('diLep', ak.to_numpy(ak.num(good_dileptons) >= 1))
        #print('tot events', nEvents, '  sum(two_lep)', ak.sum(two_lep),
        #      '\n   selection diLep', ak.sum(selection.all('diLep')), selection.all('diLep'))
        #print(ak.sum(selection.all('diLep')), ak.sum(selection.all('lumi', 'metfilter', 'diLep')))


        ll_candidates = ak.zip({
            "lep1" : good_dileptons.lep1,
            "lep2" : good_dileptons.lep2,
            "pt": (good_dileptons.lep1+good_dileptons.lep2).pt,
            "eta": (good_dileptons.lep1+good_dileptons.lep2).eta,
            "phi": (good_dileptons.lep1+good_dileptons.lep2).phi,
            "mass": (good_dileptons.lep1+good_dileptons.lep2).mass,
            "dr": good_dileptons.lep1.delta_r(good_dileptons.lep2),
        }, with_name="PtEtaPhiMLorentzVector"
                     )

        output["ndilep"].fill(ak.num(ll_candidates), weight=weights.weight())

        ll_candidates = ak.pad_none(ll_candidates, 1, axis=1)

        z_cand = ll_candidates[:, 0]
        #vpt = (z_cand.lep1 + z_cand.lep2).pt
        vmass = (z_cand.lep1 + z_cand.lep2).mass

        lepflav_mu = ak.fill_none((z_cand.lep1.lep_flav==13) | (z_cand.lep2.lep_flav==13), False)
        lepflav_el = ak.fill_none((z_cand.lep1.lep_flav==11) | (z_cand.lep2.lep_flav==11), False)
        #lepflav = lepflav_mu*1 + lepflav_el*2
        lepflav = np.array(['mu' if x&~y else 'el' if y&~x else 'emu' for x,y in zip(lepflav_mu, lepflav_el)])

        # Check sizes of arrays:
        #print("Len z cand:", len(z_cand), "nEvents:", nEvents, len(trigger_mm), len(lepflav_mu))
        trigger = np.array([m if f=='mu' else e if f=='el' else m|e for f,m,e in zip(lepflav, trigger_mm, trigger_ee)])
        #print(dataset, len(lepflav), lepflav)
        selection.add('trigger', ak.to_numpy(trigger))
        del trigger_ee
        del trigger_mm
        del trigger

        
        selection_2l = selection.all("lumi", "trigger", "metfilter", "diLep")
        #selection_2l = selection.all("lumi", "diLep")
        output["npv0"].fill(npv=events[selection_2l].PV.npvs, lepflav=lepflav[selection_2l], weight=weights.weight()[selection_2l])
        
        if self.isCorr and not isRealData:
            #print("NPVs:", events.PV.npvs)
            #print('self.SF_map', self.SF_map)
            if "PU" in self.SF_map.keys():
                weights.add("puweight", puwei(self.SF_map, events.Pileup.nTrueInt),
                            puwei(self.SF_map, events.Pileup.nTrueInt, "up"),
                            puwei(self.SF_map, events.Pileup.nTrueInt, "down") )

            #muSFs(z_cand.lep1, self.SF_map, weights, syst=self.systematics["weights"])  
            #muSFs(z_cand.lep2, self.SF_map, weights, syst=self.systematics["weights"])  
            #eleSFs(z_cand.lep1, self.SF_map, weights, syst=self.systematics["weights"])  
            #eleSFs(z_cand.lep2, self.SF_map, weights, syst=self.systematics["weights"])  

        output["npv1"].fill(npv=events[selection_2l].PV.npvs, lepflav=lepflav[selection_2l], weight=weights.weight()[selection_2l])

        #if self.isCorr and not isRealData and "LSF" in self.cfg.weights_config.keys():
        #    print('Applying LSF weights:  not working')


        jetsel = ak.fill_none(
            (events.Jet.pt > 25)
            & (abs(events.Jet.eta) <= 2.4)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 5)
            & (events.Jet.delta_r(z_cand.lep1) > 0.5) & (events.Jet.delta_r(z_cand.lep2) > 0.5), True)

        #print('z_cand:', z_cand.lep1.pt, '\n deltar:', events.Jet.delta_r(z_cand.lep1) > 0.5)
        #njet = ak.sum(jetsel, axis=1)

        good_jets = events.Jet[jetsel]

        selection.add('diJet',ak.to_numpy(ak.num(good_jets) >= 2))

        selection_2l2j = selection.all("lumi", "trigger", "metfilter", "diLep", "diJet")

        #print("NJet", nEvents, len(ak.num(good_jets)), len(ak.num(good_jets[selection_2l])), ak.sum(selection_2l), "vmass:", len(vmass[selection_2l]))

        #selection_2l_notrig = selection.all("lumi", "metfilter", "diLep")
        #print(dataset, nEvents, ak.sum(selection_2l_notrig),  ak.sum(selection_2l),  ak.sum(selection_2l2j))
        #print("\n")
        #events_2l = events[selection_2l]
        #events_2l2j = events[selection_2l2j]

        output["cutflow"][dataset]["events_2l"] += len(events[selection_2l])
        output["cutflow"][dataset]["events_2l2j"] += len(events[selection_2l2j])

        output["njet25"].fill(lepflav=lepflav[selection_2l], njet25=ak.num(good_jets[selection_2l]), weight=weights.weight()[selection_2l])

        output["dilep_m"].fill(lepflav=lepflav[selection_2l], dilep_m=vmass[selection_2l], weight=weights.weight()[selection_2l])
        output["dilep_pt"].fill(lepflav=lepflav[selection_2l], dilep_pt=z_cand[selection_2l].pt, weight=weights.weight()[selection_2l])
        output["dilep_dr"].fill(lepflav=lepflav[selection_2l], dilep_dr=z_cand[selection_2l].dr, weight=weights.weight()[selection_2l] )


        good_jets = good_jets[selection_2l2j]

        good_jets = good_jets[ak.argsort(good_jets.pt, axis=1, ascending=False)]

        dijet = good_jets[:, 0] + good_jets[:, 1]
        if isRealData:
            dijetflav = np.zeros(len(good_jets[:,0]))
        else:
            dijetflav = good_jets[:,0].hadronFlavour + good_jets[:,1].hadronFlavour
        #dijetflav = good_jets[:,0].hadronFlavour + 1*( (good_jets[:,0].partonFlavour == 0) & (good_jets[:,0].hadronFlavour == 0)) +\
        #            good_jets[:,1].hadronFlavour + 1*( (good_jets[:,0].partonFlavour == 1) & (good_jets[:,1].hadronFlavour == 0))


        jetflav = np.array(['jj' if f==0 else 'cj' if f==4 else 'bj' if f==5 else \
                            'cc' if f==8 else 'cb' if f==9 else 'bb' if f==10 else 'oo' for f in dijetflav])
        #print(dataset, len(dijetflav), len(jetflav),  '\n', dijetflav, '\n', jetflav[0:30])

        dijet_pt = dijet.pt
        dijet_m  = dijet.mass
        dijet_dr = good_jets[:, 0].delta_r(good_jets[:, 1])

        #print(len(lepflav[selection_2l2j]), len(jetflav), len(dileptons[selection_2l2j].lep1.pt), len(good_jets[:, 0]), len(weights.weight()[selection_2l2j]))

        output["lep1_pt"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav,
                               pt=z_cand[selection_2l2j].lep1.pt, weight=weights.weight()[selection_2l2j] )
        output["lep2_pt"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav,
                               pt=z_cand[selection_2l2j].lep2.pt, weight=weights.weight()[selection_2l2j] )
        output["jet1_pt"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav,
                               pt=good_jets[:, 0].pt, weight=weights.weight()[selection_2l2j] )
        output["jet2_pt"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav,
                               pt=good_jets[:, 1].pt, weight=weights.weight()[selection_2l2j] )

        dRj1 =  np.minimum( good_jets[:,0].delta_r(z_cand[selection_2l2j].lep1), good_jets[:,0].delta_r(z_cand[selection_2l2j].lep2))
        dRj2 =  np.minimum( good_jets[:,1].delta_r(z_cand[selection_2l2j].lep1), good_jets[:,1].delta_r(z_cand[selection_2l2j].lep2))

        output["jet1_dRlep"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dr=dRj1, weight=weights.weight()[selection_2l2j] )
        output["jet2_dRlep"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dr=dRj2, weight=weights.weight()[selection_2l2j] )

        output["dijet_dr"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dijet_dr=dijet_dr, weight=weights.weight()[selection_2l2j])
        output["dijet_m"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dijet_m=dijet_m, weight=weights.weight()[selection_2l2j])
        output["dijet_pt"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dijet_pt=dijet_pt, weight=weights.weight()[selection_2l2j])

        output["dijet_dr_mjj"].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav, dijet_dr=dijet_dr, dijet_mBin=dijet_m, weight=weights.weight()[selection_2l2j])

        ##print("Negative DRs:", dijet_dr[weight<0])
        ##print("Negative wei:", weight[weight<0])
        # neg_wei = np.abs(weights.weight()[selection_2l2j][weights.weight()[selection_2l2j]<0])
        # neg_wei_dr = dijet_dr[weights.weight()[selection_2l2j]<0]
        # output['dijet_dr_neg'].fill(lepflav=lepflav[selection_2l2j], jetflav=jetflav[], dijet_dr=neg_wei_dr, weight=neg_wei)

        '''
        met = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                "energy": events.MET.sumEt,
            },
            with_name="PtEtaPhiELorentzVector",
        )
        '''
        #print(dataset, "Output", psutil.Process(getpid()).memory_info().rss / 1024 ** 2, "MB")
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator


def main():
    print("There is no main() usage of this script")


if __name__ == "__main__":
    main()
