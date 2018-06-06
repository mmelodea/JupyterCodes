###------------------------------------------------------------------------------------###
###  This code runs over ROOT files and creates a dictionary for the variables        ###
###  Its dictionary is used then by EvaluateNeuralNetwork.py                           ###
###  Author: Miqueias Melo de Almeida                                                  ###
###  Data: 05/06/2018                                                                  ###
###------------------------------------------------------------------------------------###

#set adequate environment
import os
import sys
import argparse
import cPickle as pickle
import ROOT
import math



def FormatInputs(list_of_files, output_file, tags, keys, Nevents, Njets, MaxSubjets, Nsubjets):
  file_names = open(list_of_files,'r')
  files_addresses = []
  print 'Retrieving files addresses...'
  for i in file_names.readlines():
    files_addresses.append( i.rstrip()  )


  #format inputs from TTree (Data & MC)
  default_value = 0
  formated_inputs = {}
  mc_sumweight = {}
  event_index = {}
  print 'Creating dictionary...'
  for ik in keys:
    if ik in formated_inputs:
      continue
    
    formated_inputs[ik] = {}
    mc_sumweight[ik] = 0
    event_index[ik] = 0
    
    #instantiate dictionary
    formated_inputs[ik]['mc'] = []
    formated_inputs[ik]['lowStat'] = []
    formated_inputs[ik]['final_state'] = []
    formated_inputs[ik]['mc_sumweight'] = []
    formated_inputs[ik]['f_outlier'] = []
    formated_inputs[ik]['f_run'] = []
    formated_inputs[ik]['f_lumi'] = []
    formated_inputs[ik]['f_event'] = []
    formated_inputs[ik]['f_weight'] = []
    formated_inputs[ik]['f_Djet_VAJHU'] = []
    formated_inputs[ik]['f_Djet_VAJHU_UncDn'] = []
    formated_inputs[ik]['f_Djet_VAJHU_UncUp'] = []
    formated_inputs[ik]['f_lept1_pt'] = []
    formated_inputs[ik]['f_lept1_pt_error'] = []
    formated_inputs[ik]['f_lept1_eta'] = []
    formated_inputs[ik]['f_lept1_phi'] = []
    formated_inputs[ik]['f_lept1_charge'] = []
    formated_inputs[ik]['f_lept1_pfx'] = []
    formated_inputs[ik]['f_lept1_sip'] = []
    formated_inputs[ik]['f_lept1_pdgid'] = []
    formated_inputs[ik]['f_lept2_pt'] = []
    formated_inputs[ik]['f_lept2_pt_error'] = []
    formated_inputs[ik]['f_lept2_eta'] = []
    formated_inputs[ik]['f_lept2_phi'] = []
    formated_inputs[ik]['f_lept2_charge'] = []
    formated_inputs[ik]['f_lept2_pfx'] = []
    formated_inputs[ik]['f_lept2_sip'] = []
    formated_inputs[ik]['f_lept2_pdgid'] = []
    formated_inputs[ik]['f_lept3_pt'] = []
    formated_inputs[ik]['f_lept3_pt_error'] = []
    formated_inputs[ik]['f_lept3_eta'] = []
    formated_inputs[ik]['f_lept3_phi'] = []
    formated_inputs[ik]['f_lept3_charge'] = []
    formated_inputs[ik]['f_lept3_pfx'] = []
    formated_inputs[ik]['f_lept3_sip'] = []
    formated_inputs[ik]['f_lept3_pdgid'] = []
    formated_inputs[ik]['f_lept4_pt'] = []
    formated_inputs[ik]['f_lept4_pt_error'] = []
    formated_inputs[ik]['f_lept4_eta'] = []
    formated_inputs[ik]['f_lept4_phi'] = []
    formated_inputs[ik]['f_lept4_charge'] = []
    formated_inputs[ik]['f_lept4_pfx'] = []
    formated_inputs[ik]['f_lept4_sip'] = []
    formated_inputs[ik]['f_lept4_pdgid'] = []    
    formated_inputs[ik]['f_iso_max'] = []
    formated_inputs[ik]['f_sip_max'] = []
    formated_inputs[ik]['f_Z1mass'] = []
    formated_inputs[ik]['f_Z2mass'] = []
    formated_inputs[ik]['f_angle_costhetastar'] = []
    formated_inputs[ik]['f_angle_costheta1'] = []
    formated_inputs[ik]['f_angle_costheta2'] = []
    formated_inputs[ik]['f_angle_phi'] = []
    formated_inputs[ik]['f_angle_phistar1'] = []
    formated_inputs[ik]['f_pt4l'] = []
    formated_inputs[ik]['f_eta4l'] = []
    formated_inputs[ik]['f_mass4l'] = []
    formated_inputs[ik]['f_njets_pass'] = []
    formated_inputs[ik]['f_deltajj'] = []
    formated_inputs[ik]['f_massjj'] = []
    formated_inputs[ik]['f_pfmet'] = []
    formated_inputs[ik]['f_pfmet_eta'] = []
    formated_inputs[ik]['f_pfmet_phi'] = []
    formated_inputs[ik]['f_pfmet_JetEnUp'] = []
    formated_inputs[ik]['f_pfmet_JetEnDn'] = []
    formated_inputs[ik]['f_pfmet_JetResUp'] = []
    formated_inputs[ik]['f_pfmet_JetResDn'] = []
    formated_inputs[ik]['f_pfmet_ElectronEnUp'] = []
    formated_inputs[ik]['f_pfmet_ElectronEnDn'] = []
    formated_inputs[ik]['f_pfmet_MuonEnUp'] = []
    formated_inputs[ik]['f_pfmet_MuonEnDn'] = []
    formated_inputs[ik]['f_pfmet_UnclusteredEnUp'] = []
    formated_inputs[ik]['f_pfmet_UnclusteredEnDn'] = []
    formated_inputs[ik]['f_pfmet_PhotonEnUp'] = []
    formated_inputs[ik]['f_pfmet_PhotonEnDn'] = []
    formated_inputs[ik]['f_mT'] = []
    formated_inputs[ik]['f_dphi'] = []
    formated_inputs[ik]['f_nbjets'] = []
    for i in range(Njets):
      #formated_inputs[ik]['f_jets_highpt_btagger[%i]' % i] = []
      formated_inputs[ik]['f_jets_highpt_pt[%i]' % i] = []
      formated_inputs[ik]['f_jets_highpt_pt_error[%i]' % i] = []
      formated_inputs[ik]['f_jets_highpt_eta[%i]' % i] = []
      formated_inputs[ik]['f_jets_highpt_phi[%i]' % i] = []
      formated_inputs[ik]['f_jets_highpt_e[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_area[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_ptd[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_charged_hadron_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_neutral_hadron_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_photon_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_electron_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_muon_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_hf_hadron_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_hf_em_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_charged_em_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_charged_mu_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_neutral_em_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_charged_hadron_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_neutral_hadron_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_photon_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_electron_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_muon_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_hf_hadron_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_hf_em_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_charged_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_neutral_multiplicity[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_ncomponents[%i]' % i] = []
      #### each of this guys will have maxJets objects
      #formated_inputs[ik]['f_jets_highpt_component_pdgid[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_pt[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_eta[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_phi[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_energy[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_charge[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_mt[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_xvertex[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_yvertex[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_zvertex[%i]' % i] = []
      #formated_inputs[ik]['f_jets_highpt_component_vertex_chi2[%i]' % i] = []
    
    
  print 'Starting to load files...'
  ntags = len(tags)
  for ifile in range(len(files_addresses)):
    file_name = files_addresses[ifile]
    
    imc = ''
    for itag in range(ntags):
      if(file_name.find(tags[itag]) != -1):
	imc = keys[itag]
      else:
	continue
    
    if(imc not in keys):
      continue
  
    tfile = ROOT.TFile.Open(file_name)
    print 'Processing %s' % tfile.GetName()
    tree = tfile.Get('HZZ4LeptonsAnalysisReduced')
    tree.GetEntries()
      
    for ievt, evt in enumerate(tree):
      #VBF category, -MELA cut, +m4l cut
      if( (((evt.f_njets_pass == 2 or evt.f_njets_pass == 3) and evt.f_Nbjets <= 1) or (evt.f_njets_pass > 3 and evt.f_Nbjets == 0)) and evt.f_mass4l >= 118 and evt.f_mass4l <= 130 ):
	#### use Njets to split in categories
	if(evt.f_njets_pass < Njets):
	  continue
            
        #gets the sum of weights
        event_index[imc] += 1
	if(event_index[imc] > Nevents):
          break;

        mc_sumweight[imc] += evt.f_weight

        formated_inputs[imc]['mc'].append( imc )#0
        if(file_name.find('histos4mu') != -1):
            formated_inputs[imc]['final_state'].append( '4mu' )
        if(file_name.find('histos4e') != -1):
            formated_inputs[imc]['final_state'].append( '4e' )
        if(file_name.find('histos2e2mu') != -1):
            formated_inputs[imc]['final_state'].append( '2e2mu' )

	formated_inputs[imc]['lowStat'].append( 0 )
	formated_inputs[imc]['f_outlier'].append( evt.f_outlier )
	formated_inputs[imc]['f_run'].append( evt.f_run )
	formated_inputs[imc]['f_lumi'].append( evt.f_lumi )
	formated_inputs[imc]['f_event'].append( evt.f_event )
	formated_inputs[imc]['mc_sumweight'].append( 0 )
	formated_inputs[imc]['f_weight'].append( evt.f_weight )
	formated_inputs[imc]['f_Djet_VAJHU'].append( evt.f_Djet_VAJHU )
	formated_inputs[imc]['f_Djet_VAJHU_UncUp'].append( evt.f_Djet_VAJHU_UncUp )
	formated_inputs[imc]['f_Djet_VAJHU_UncDn'].append( evt.f_Djet_VAJHU_UncDn )
	formated_inputs[imc]['f_lept1_pt'].append( evt.f_lept1_pt )
	formated_inputs[imc]['f_lept1_pt_error'].append( evt.f_lept1_pt_error )
	formated_inputs[imc]['f_lept1_eta'].append( evt.f_lept1_eta )
	formated_inputs[imc]['f_lept1_phi'].append( evt.f_lept1_phi )
	formated_inputs[imc]['f_lept1_charge'].append( evt.f_lept1_charge )
	formated_inputs[imc]['f_lept1_pfx'].append( evt.f_lept1_pfx )
	formated_inputs[imc]['f_lept1_sip'].append( evt.f_lept1_sip )
	formated_inputs[imc]['f_lept1_pdgid'].append( evt.f_lept1_pdgid )
	formated_inputs[imc]['f_lept2_pt'].append( evt.f_lept2_pt )
	formated_inputs[imc]['f_lept2_pt_error'].append( evt.f_lept2_pt_error )
	formated_inputs[imc]['f_lept2_eta'].append( evt.f_lept2_eta )
	formated_inputs[imc]['f_lept2_phi'].append( evt.f_lept2_phi )
	formated_inputs[imc]['f_lept2_charge'].append( evt.f_lept2_charge )
	formated_inputs[imc]['f_lept2_pfx'].append( evt.f_lept2_pfx )
	formated_inputs[imc]['f_lept2_sip'].append( evt.f_lept2_sip )
	formated_inputs[imc]['f_lept2_pdgid'].append( evt.f_lept2_pdgid )
	formated_inputs[imc]['f_lept3_pt'].append( evt.f_lept3_pt )
	formated_inputs[imc]['f_lept3_pt_error'].append( evt.f_lept3_pt_error )
	formated_inputs[imc]['f_lept3_eta'].append( evt.f_lept3_eta )
	formated_inputs[imc]['f_lept3_phi'].append( evt.f_lept3_phi )
	formated_inputs[imc]['f_lept3_charge'].append( evt.f_lept3_charge )
	formated_inputs[imc]['f_lept3_pfx'].append( evt.f_lept3_pfx )
	formated_inputs[imc]['f_lept3_sip'].append( evt.f_lept3_sip )
	formated_inputs[imc]['f_lept3_pdgid'].append( evt.f_lept3_pdgid )
	formated_inputs[imc]['f_lept4_pt'].append( evt.f_lept4_pt )
	formated_inputs[imc]['f_lept4_pt_error'].append( evt.f_lept4_pt_error )
	formated_inputs[imc]['f_lept4_eta'].append( evt.f_lept4_eta )
	formated_inputs[imc]['f_lept4_phi'].append( evt.f_lept4_phi )
	formated_inputs[imc]['f_lept4_charge'].append( evt.f_lept4_charge )
	formated_inputs[imc]['f_lept4_pfx'].append( evt.f_lept4_pfx )
	formated_inputs[imc]['f_lept4_sip'].append( evt.f_lept4_sip )
	formated_inputs[imc]['f_lept4_pdgid'].append( evt.f_lept4_pdgid )    
	formated_inputs[imc]['f_iso_max'].append( evt.f_iso_max )
	formated_inputs[imc]['f_sip_max'].append( evt.f_sip_max )
	formated_inputs[imc]['f_Z1mass'].append( evt.f_Z1mass )
	formated_inputs[imc]['f_Z2mass'].append( evt.f_Z2mass )
	formated_inputs[imc]['f_angle_costhetastar'].append( evt.f_angle_costhetastar )
	formated_inputs[imc]['f_angle_costheta1'].append( evt.f_angle_costheta1 )
	formated_inputs[imc]['f_angle_costheta2'].append( evt.f_angle_costheta2 )
	formated_inputs[imc]['f_angle_phi'].append( evt.f_angle_phi )
	formated_inputs[imc]['f_angle_phistar1'].append( evt.f_angle_phistar1 )
	formated_inputs[imc]['f_pt4l'].append( evt.f_pt4l )
	formated_inputs[imc]['f_eta4l'].append( evt.f_eta4l )
	formated_inputs[imc]['f_mass4l'].append( evt.f_mass4l )
	formated_inputs[imc]['f_njets_pass'].append( evt.f_njets_pass )
	formated_inputs[imc]['f_deltajj'].append( evt.f_deltajj )
	formated_inputs[imc]['f_massjj'].append( evt.f_massjj )
	formated_inputs[imc]['f_pfmet'].append( evt.f_pfmet )
        formated_inputs[imc]['f_pfmet_eta'].append( -ROOT.TMath.Log(ROOT.TMath.Tan(evt.f_pfmet_theta/2.)) )
        formated_inputs[imc]['f_pfmet_phi'].append( evt.f_pfmet_phi )
	formated_inputs[imc]['f_pfmet_JetEnUp'].append( evt.f_pfmet_JetEnUp )
	formated_inputs[imc]['f_pfmet_JetEnDn'].append( evt.f_pfmet_JetEnDn )
	formated_inputs[imc]['f_pfmet_JetResUp'].append( evt.f_pfmet_JetResUp )
	formated_inputs[imc]['f_pfmet_JetResDn'].append( evt.f_pfmet_JetResDn )
	formated_inputs[imc]['f_pfmet_ElectronEnUp'].append( evt.f_pfmet_ElectronEnUp )
	formated_inputs[imc]['f_pfmet_ElectronEnDn'].append( evt.f_pfmet_ElectronEnDn )
	formated_inputs[imc]['f_pfmet_MuonEnUp'].append( evt.f_pfmet_MuonEnUp )
	formated_inputs[imc]['f_pfmet_MuonEnDn'].append( evt.f_pfmet_MuonEnDn )
	formated_inputs[imc]['f_pfmet_UnclusteredEnUp'].append( evt.f_pfmet_UnclusteredEnUp )
	formated_inputs[imc]['f_pfmet_UnclusteredEnDn'].append( evt.f_pfmet_UnclusteredEnDn )
	formated_inputs[imc]['f_pfmet_PhotonEnUp'].append( evt.f_pfmet_PhotonEnUp )
	formated_inputs[imc]['f_pfmet_PhotonEnDn'].append( evt.f_pfmet_PhotonEnDn )
	formated_inputs[imc]['f_mT'].append( evt.f_mT )
	formated_inputs[imc]['f_dphi'].append( evt.f_dphi )
	formated_inputs[imc]['f_nbjets'].append( evt.f_Nbjets )
	for ijet in range(Njets):
	  check = evt.f_jets_highpt_pt[ijet]
	  #formated_inputs[imc]['f_jets_highpt_btagger[%i]' % ijet].append( (evt.f_jets_highpt_btagger[ijet]  if (check != -999) else default_value) )
	  formated_inputs[imc]['f_jets_highpt_pt[%i]' % ijet].append( (evt.f_jets_highpt_pt[ijet]  if (check != -999) else default_value) )
	  formated_inputs[imc]['f_jets_highpt_pt_error[%i]' % ijet].append( (evt.f_jets_highpt_pt_error[ijet]  if (check != -999) else default_value) )
	  formated_inputs[imc]['f_jets_highpt_eta[%i]' % ijet].append( (evt.f_jets_highpt_eta[ijet]  if (check != -999) else default_value) )
	  formated_inputs[imc]['f_jets_highpt_phi[%i]' % ijet].append( (evt.f_jets_highpt_phi[ijet]  if (check != -999) else default_value) )
	  jetE = evt.f_jets_highpt_et[ijet]*ROOT.TMath.CosH(evt.f_jets_highpt_eta[ijet])
	  formated_inputs[imc]['f_jets_highpt_e[%i]' % ijet].append( (jetE if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_area[%i]' % ijet].append( (evt.f_jets_highpt_area[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_ptd[%i]' % ijet].append( (evt.f_jets_highpt_ptd[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_charged_hadron_energy[%i]' % ijet].append( (evt.f_jets_highpt_charged_hadron_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_neutral_hadron_energy[%i]' % ijet].append( (evt.f_jets_highpt_neutral_hadron_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_photon_energy[%i]' % ijet].append( (evt.f_jets_highpt_photon_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_electron_energy[%i]' % ijet].append( (evt.f_jets_highpt_electron_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_muon_energy[%i]' % ijet].append( (evt.f_jets_highpt_muon_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_hf_hadron_energy[%i]' % ijet].append( (evt.f_jets_highpt_hf_hadron_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_hf_em_energy[%i]' % ijet].append( (evt.f_jets_highpt_hf_em_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_charged_em_energy[%i]' % ijet].append( (evt.f_jets_highpt_charged_em_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_charged_mu_energy[%i]' % ijet].append( (evt.f_jets_highpt_charged_mu_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_neutral_em_energy[%i]' % ijet].append( (evt.f_jets_highpt_neutral_em_energy[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_charged_hadron_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_charged_hadron_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_neutral_hadron_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_neutral_hadron_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_photon_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_photon_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_electron_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_electron_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_muon_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_muon_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_hf_hadron_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_hf_hadron_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_hf_em_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_hf_em_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_charged_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_charged_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_neutral_multiplicity[%i]' % ijet].append( (evt.f_jets_highpt_neutral_multiplicity[ijet]  if (check != -999) else default_value) )
	  #formated_inputs[imc]['f_jets_highpt_ncomponents[%i]' % ijet].append( (evt.f_jets_highpt_ncomponents[ijet]  if (check != -999) else default_value) )
	  #### each of this guys will have 100 objects
	  #vecs = [[],[],[],[],[],[],[],[],[],[],[]]
	  #count_jets = 0
	  #for isjet in range(ijet*MaxSubjets,(ijet+1)*MaxSubjets):
	  #  check = evt.f_jets_highpt_component_pt[isjet]
	  #  vecs[0]  = (evt.f_jets_highpt_component_pdgid[isjet]       if (check != -999) else default_value)
	  #  vecs[1]  = (evt.f_jets_highpt_component_pt[isjet]          if (check != -999) else default_value)
	  #  vecs[2]  = (evt.f_jets_highpt_component_eta[isjet]         if (check != -999) else default_value)
	  #  vecs[3]  = (evt.f_jets_highpt_component_phi[isjet]         if (check != -999) else default_value)
	  #  vecs[4]  = (evt.f_jets_highpt_component_energy[isjet]      if (check != -999) else default_value)
	  #  vecs[5]  = (evt.f_jets_highpt_component_charge[isjet]      if (check != -999) else default_value)
	  #  vecs[6]  = (evt.f_jets_highpt_component_mt[isjet]          if (check != -999) else default_value)
	  #  vecs[7]  = (evt.f_jets_highpt_component_xvertex[isjet]     if (check != -999) else default_value)
	  #  vecs[8]  = (evt.f_jets_highpt_component_yvertex[isjet]     if (check != -999) else default_value)
	  #  vecs[9]  = (evt.f_jets_highpt_component_zvertex[isjet]     if (check != -999) else default_value)
	  #  vecs[10] = (evt.f_jets_highpt_component_vertex_chi2[isjet] if (check != -999) else default_value)
	  #  count_jets += 1
	  #  if(count_jets > Nsubjets):
	  #    break
	     
	  #formated_inputs[imc]['f_jets_highpt_component_pdgid[%i]' % ijet].append( vecs[0] )
	  #formated_inputs[imc]['f_jets_highpt_component_pt[%i]' % ijet].append( vecs[1] )
	  #formated_inputs[imc]['f_jets_highpt_component_eta[%i]' % ijet].append( vecs[2] )
	  #formated_inputs[imc]['f_jets_highpt_component_phi[%i]' % ijet].append( vecs[3] )
	  #formated_inputs[imc]['f_jets_highpt_component_energy[%i]' % ijet].append( vecs[4] )
	  #formated_inputs[imc]['f_jets_highpt_component_charge[%i]' % ijet].append( vecs[5] )
	  #formated_inputs[imc]['f_jets_highpt_component_mt[%i]' % ijet].append( vecs[6] )
	  #formated_inputs[imc]['f_jets_highpt_component_xvertex[%i]' % ijet].append( vecs[7] )
	  #formated_inputs[imc]['f_jets_highpt_component_yvertex[%i]' % ijet].append( vecs[8] )
	  #formated_inputs[imc]['f_jets_highpt_component_zvertex[%i]' % ijet].append( vecs[9] )
	  #formated_inputs[imc]['f_jets_highpt_component_vertex_chi2[%i]' % ijet].append( vecs[10] )

    tfile.Close()
    #print ('%i - Processed (%i)' % (ifile, nacepted)), file_name
    
  print ''
  print '------------ Final Dataset Organization ----------------'
  print '--------------------- MCs ------------------------------'
  fkeys = formated_inputs.keys()
  print fkeys
  print '----------------- Variables ----------------------------'
  print formated_inputs[fkeys[0]].keys()


  #for randomizing FS composition in samples
  print ''
  print '>>>>>>>>>> Removing empty keys, Flagging low statistic and Shuffling events...'
  rmkeys = []
  for ik in formated_inputs:
    if len(formated_inputs[ik]['mc_sumweight']) == 0:
      print '%s has no events... will be removed!' % ik
      rmkeys.append(ik)
  for ik in rmkeys:
    del formated_inputs[ik]

  import random
  for ik in formated_inputs:
    #inserts the correct mc weight
    nevents = len(formated_inputs[ik]['mc_sumweight'])
    for iev in range(nevents):
      formated_inputs[ik]['mc_sumweight'][iev] = mc_sumweight[ik]
      if(nevents < 60):
	formated_inputs[ik]['lowStat'][iev] = 1

    print ik," -- nevents = ",event_index[ik]," -- sumweight = ",mc_sumweight[ik]
    for ivar in formated_inputs[ik]:
      #shuffle twice to get more randomization
      random.seed(999) #keeping the same seed -> events ordering is preserved
      random.shuffle(formated_inputs[ik][ivar])
      random.seed(348) #keeping the same seed -> events ordering is preserved
      random.shuffle(formated_inputs[ik][ivar])
        
    
  #save the dictionary
  print ''
  print 'Saving dataset...'
  fileout = open(output_file+'.pkl','w')
  pickle.dump( formated_inputs, fileout )
  fileout.close()





def main(options):
  print '>>>>>>>>>> Inputs <<<<<<<<<<<<<'
  print options.infile
  print options.outfile
  print options.tags
  print options.keys
  print options.nevents
  print options.njets
  print options.maxsubjets
  print options.nsubjets
  
  if(len(options.keys)==len(options.tags)):
    print '>>>>>>>>>>  Assigments <<<<<<<<<<'
    for key,tag in zip(options.keys,options.tags):
      print key,' -- ',tag
    print ''
    FormatInputs(options.infile, options.outfile, options.tags, options.keys, options.nevents, options.njets, options.maxsubjets, options.nsubjets)
  else:
    print 'ERROR: Incompatible length of keys and tags'
  
  
  
if __name__ == '__main__':

 # Setup argument parser
 parser = argparse.ArgumentParser()

 # Add more arguments
 parser.add_argument("--infile", help="Name of txt input file with addresses of root files")
 parser.add_argument("--outfile", help="Name of the output file")
 parser.add_argument("--tags", nargs='+', help="Tags to find the MCs (from the rootfiles) to be used")
 parser.add_argument("--keys", nargs='+', help="Tags for the MCs to be used inside Keras framework")
 parser.add_argument("--nevents", type=int, default=10000000, help="Number of events to get from each file")
 parser.add_argument("--njets", type=int, default=4, help="Number of jets to be picked")
 parser.add_argument("--maxsubjets", type=int, default=10, help="Number of jet components to be picked")
 parser.add_argument("--nsubjets", type=int, default=10, help="Number of jet components to be picked")

 # Parse default arguments
 options = parser.parse_args()
 main(options)
    
