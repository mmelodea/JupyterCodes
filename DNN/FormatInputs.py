import os
import sys
import cPickle as pickle
import ROOT

#loads input data
#path = '/home/micah/temp/hzz4l_histos/OrganizedSamples/'
#file_names4e = open(path+'histos4e_25ns/file_names.txt','r')
#file_names4mu = open(path+'histos4mu_25ns/file_names.txt','r')
#file_names2e2mu = open(path+'histos2e2mu_25ns/file_names.txt','r')

path = '/home/micah/temp/hzz4l_histos/OrganizedSamples2/'
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
default_value = 0
formated_inputs = {}
#keys = ['Data','VBF','HJJ','ttH','ZH','WH','ggZZ','qqZZ','ttZ','ggH','myqqZZ']
keys = ['VBF','HJJ','ttH','ZH','WH','qqZZ']
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
    
    if(imc not in keys):
        continue
  
    tfile = ROOT.TFile.Open(file_name)
    print 'Processing %s' % tfile.GetName()
    tree = tfile.Get('HZZ4LeptonsAnalysisReduced')
    tree.GetEntries()
      
    for ievt, evt in enumerate(tree):
      #VBF category, -Djet, +m4l cut
      if( (((evt.f_njets_pass == 2 or evt.f_njets_pass == 3) and evt.f_Nbjets <= 1) or (evt.f_njets_pass > 3 and evt.f_Nbjets == 0)) and evt.f_mass4l >= 118 and evt.f_mass4l <= 130 ):
            
        event = []        
        event.append( evt.f_weight )#0
        event.append( evt.f_D_jet )#1
        event.append( evt.f_Djet_VAJHU )#2
        event.append( evt.f_lept1_pt )#3
        event.append( evt.f_lept1_eta )#4
        event.append( evt.f_lept1_phi )#5
        event.append( evt.f_lept1_charge )#6
        event.append( evt.f_lept1_pfx )#7
        event.append( evt.f_lept1_sip )#8
        event.append( evt.f_lept1_pdgid )#9
        event.append( evt.f_lept2_pt )#10
        event.append( evt.f_lept2_eta )#11
        event.append( evt.f_lept2_phi )#12
        event.append( evt.f_lept2_charge )#13
        event.append( evt.f_lept2_pfx )#14
        event.append( evt.f_lept2_sip )#15
        event.append( evt.f_lept2_pdgid )#16
        event.append( evt.f_lept3_pt )#17
        event.append( evt.f_lept3_eta )#18
        event.append( evt.f_lept3_phi )#19
        event.append( evt.f_lept3_charge )#20
        event.append( evt.f_lept3_pfx )#21
        event.append( evt.f_lept3_sip )#22
        event.append( evt.f_lept3_pdgid )#23
        event.append( evt.f_lept4_pt )#24
        event.append( evt.f_lept4_eta )#25
        event.append( evt.f_lept4_phi )#26
        event.append( evt.f_lept4_charge )#27
        event.append( evt.f_lept4_pfx )#28
        event.append( evt.f_lept4_sip )#29
        event.append( evt.f_lept4_pdgid )#30
        event.append( evt.f_iso_max )#31
        event.append( evt.f_sip_max )#32
        event.append( evt.f_Z1mass )#33
        event.append( evt.f_Z2mass )#34
        event.append( evt.f_angle_costhetastar )#35
        event.append( evt.f_angle_costheta1 )#36
        event.append( evt.f_angle_costheta2 )#37
        event.append( evt.f_angle_phi )#38
        event.append( evt.f_angle_phistar1 )#39
        event.append( evt.f_pt4l )#40
        event.append( evt.f_eta4l )#41
        event.append( evt.f_mass4l )#42
        event.append( evt.f_njets_pass )#43
        event.append( evt.f_deltajj )#44
        event.append( evt.f_massjj )#45
        for ijet in range(3):
            if(evt.f_jets_dnn_pt[ijet] != -999):
                event.append( evt.f_jets_dnn_pt[ijet] )#46
                event.append( evt.f_jets_dnn_eta[ijet] )#47
                event.append( evt.f_jets_dnn_phi[ijet] )#48
                event.append( evt.f_jets_dnn_e[ijet] )#49
                event.append( evt.f_jet_subjetness[ijet] )#50
                event.append( evt.f_jet_ptD[ijet] )#51
                event.append( evt.f_jet_photonEnergy[ijet] )#52
                event.append( evt.f_jet_electronEnergy[ijet] )#53
                event.append( evt.f_jet_muonEnergy[ijet] )#54
                event.append( evt.f_jet_chargedEmEnergy[ijet] )#55
                event.append( evt.f_jet_neutralEmEnergy[ijet] )#56
                event.append( evt.f_jet_chargedHadronEnergy[ijet] )#57
                event.append( evt.f_jet_neutralHadronEnergy[ijet] )#58
                for isjet in range(ijet*100, (ijet+1)*100):#59-359
                    if(evt.f_jet_component_pt[isjet] != -999):
                        event.append( evt.f_jet_component_pt[isjet] )  
                        event.append( evt.f_jet_component_eta[isjet] )  
                        event.append( evt.f_jet_component_phi[isjet] )
                    else:
                        event.append( default_value )  
                        event.append( default_value )  
                        event.append( default_value )
                        
            else:
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                event.append( default_value )
                for isjet in range(100):
                    event.append( default_value )  
                    event.append( default_value )  
                    event.append( default_value )  
                
        event.append( evt.f_pfmet )
        event.append( evt.f_genmet )
        event.append( evt.f_mT )
        event.append( evt.f_dphi )
        event.append( evt.f_Nbjets )
      
        #--------------------------------#
        #print '----------------------------------------------------------------------------'
        #for i in range(len(event)):
        #    print i,": ",event[i]
        #break    
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
fileout = open('hzz4l_vbf_selection_noDjet_m4l118-130GeV_shuffledFS_JetExtraInfo.pkl','w')
pickle.dump( formated_inputs, fileout )
fileout.close()