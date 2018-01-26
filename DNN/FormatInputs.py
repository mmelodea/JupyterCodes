import os
import sys
import cPickle as pickle
import ROOT

#loads input data
path = '/home/micah/temp/hzz4l_histos/OrganizedSamples/'
file_names4e = open(path+'histos4e_25ns/file_names.txt','r')
file_names4mu = open(path+'histos4mu_25ns/file_names.txt','r')
file_names2e2mu = open(path+'histos2e2mu_25ns/file_names.txt','r')

files_addresses = []
for i in file_names4mu.readlines():
    files_addresses.append( i.rstrip()  )
for i in file_names4e.readlines():
    files_addresses.append( i.rstrip()  )
for i in file_names2e2mu.readlines():
    files_addresses.append( i.rstrip() )


#format inputs from TTree (Data & MC)
formated_inputs = {}
keys = ['Data','VBF','HJJ','ttH','ZH','WH','ggZZ','qqZZ','ttZ','ggH','myqqZZ']
for ik in keys:
  formated_inputs[ik] = []

for ifile in range(len(files_addresses)):
  file_name = files_addresses[ifile]
        
  imc = ''
  if(file_name.find('output_VBF') != -1):
    imc = 'VBF'
  elif(file_name.find('output_GluGluHToZZTo4L_M125_13TeV_powheg2_minloHJJ') != -1):
    imc = 'HJJ'
  elif(file_name.find('output_GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6') != -1):
    imc = 'ggH'
  elif(file_name.find('output_ttH') != -1):
    imc = 'ttH'
  elif(file_name.find('output_ZH') != -1):
    imc = 'ZH'
  elif(file_name.find('output_WminusH') != -1 or file_name.find('output_WplusH') != -1):
    imc = 'WH'
  elif(file_name.find('output_GluGluToContinToZZ') != -1):
    imc = 'ggZZ'
  elif(file_name.find('output_ZZTo4L') != -1):
    imc = 'qqZZ'
  elif(file_name.find('output_POWHEG') != -1):
    imc = 'myqqZZ'            
  elif(file_name.find('output_TTZ') != -1):
    imc = 'ttZ'
  elif(file_name.find('Run2016') != -1):
    imc = 'Data'
  else:
    continue

  tfile = ROOT.TFile.Open(file_name)
  print 'Processing %s' % tfile.GetName()
  tree = tfile.Get('HZZ4LeptonsAnalysisReduced')
  tree.GetEntries()
      
  for ievt, evt in enumerate(tree):
    #VBF category, -Djet, +m4l cut
    if( (((evt.f_njets_pass == 2 or evt.f_njets_pass == 3) and evt.f_Nbjets <= 1) or (evt.f_njets_pass > 3 and evt.f_Nbjets == 0)) and evt.f_mass4l >= 118 and evt.f_mass4l <= 130 ):
                
      event = []        
      event.append( evt.f_weight )
      event.append( evt.f_D_jet )
      event.append( evt.f_Djet_VAJHU )
      event.append( evt.f_lept1_pt )
      event.append( evt.f_lept1_eta )
      event.append( evt.f_lept1_phi )
      event.append( evt.f_lept1_charge )
      event.append( evt.f_lept1_pfx )
      event.append( evt.f_lept1_sip )
      event.append( evt.f_lept1_pdgid )
      event.append( evt.f_lept2_pt )
      event.append( evt.f_lept2_eta )
      event.append( evt.f_lept2_phi )
      event.append( evt.f_lept2_charge )
      event.append( evt.f_lept2_pfx )
      event.append( evt.f_lept2_sip )
      event.append( evt.f_lept2_pdgid )
      event.append( evt.f_lept3_pt )
      event.append( evt.f_lept3_eta )
      event.append( evt.f_lept3_phi )
      event.append( evt.f_lept3_charge )
      event.append( evt.f_lept3_pfx )
      event.append( evt.f_lept3_sip )
      event.append( evt.f_lept3_pdgid )
      event.append( evt.f_lept4_pt )
      event.append( evt.f_lept4_eta )
      event.append( evt.f_lept4_phi )
      event.append( evt.f_lept4_charge )
      event.append( evt.f_lept4_pfx )
      event.append( evt.f_lept4_sip )
      event.append( evt.f_lept4_pdgid )
      event.append( evt.f_iso_max )
      event.append( evt.f_sip_max )
      event.append( evt.f_Z1mass )
      event.append( evt.f_Z2mass )
      event.append( evt.f_angle_costhetastar )
      event.append( evt.f_angle_costheta1 )
      event.append( evt.f_angle_costheta2 )
      event.append( evt.f_angle_phi )
      event.append( evt.f_angle_phistar1 )
      event.append( evt.f_pt4l )
      event.append( evt.f_eta4l )
      event.append( evt.f_mass4l )
      event.append( evt.f_njets_pass )
      event.append( evt.f_deltajj )
      event.append( evt.f_massjj )
      event.append( evt.f_jets_dnn_pt[0] )
      event.append( evt.f_jets_dnn_eta[0] )
      event.append( evt.f_jets_dnn_phi[0] )
      event.append( evt.f_jets_dnn_e[0] )
      event.append( evt.f_jets_dnn_pt[1] )
      event.append( evt.f_jets_dnn_eta[1] )
      event.append( evt.f_jets_dnn_phi[1] )
      event.append( evt.f_jets_dnn_e[1] )
      event.append( evt.f_jets_dnn_pt[2] )
      event.append( evt.f_jets_dnn_eta[2] )
      event.append( evt.f_jets_dnn_phi[2] )
      event.append( evt.f_jets_dnn_e[2] )                
      event.append( evt.f_pfmet )
      event.append( evt.f_genmet )
      event.append( evt.f_mT )
      event.append( evt.f_dphi )
      event.append( evt.f_Nbjets )
					      
      formated_inputs[imc].append(event)
                        

  tfile.Close()
  #print ('%i - Processed (%i)' % (ifile, nacepted)), file_name


keys = []
for ik in formated_inputs:
    keys.append(ik)
print '----------- KEYS -----------'
print keys    

#for randomizing FS composition in samples
print 'shuffling events...'
import random
random.seed(999)
for ik in formated_inputs:
    random.shuffle(formated_inputs[ik])

#save the dictionary
fileout = open('hzz4l_vbf_selection_noDjet_m4l118-130GeV_shuffledFS.pkl','w')
pickle.dump( formated_inputs, fileout )
fileout.close()