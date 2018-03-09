import os
import sys
import cPickle as pickle
import ROOT
import math

#loads input data
path = '/home/micah/Documents/HiggsRunII/hzz4l_histos/OriginalSamples/'
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
mc_sumweight = {}
event_index = {}
keys = ['Data','VBF','HJJ','ttH','ZH','WH','ggZZ','qqZZ','ttZ']
#keys = ['VBF','HJJ','ttH','ZH','WH','qqZZ'] #samples with jet substructure
for ik in keys:
    formated_inputs[ik] = {}
    mc_sumweight[ik] = 0
    event_index[ik] = 0
    
    #instantiate dictionary
    formated_inputs[ik]['mc'] = []
    formated_inputs[ik]['final_state'] = []
    formated_inputs[ik]['event_index'] = []
    formated_inputs[ik]['mc_sumweight'] = []
    formated_inputs[ik]['event_weight'] = []
    formated_inputs[ik]['Djet'] = []
    formated_inputs[ik]['DjetVAJHU'] = []
    formated_inputs[ik]['l1pt'] = []
    formated_inputs[ik]['l1eta'] = []
    formated_inputs[ik]['l1phi'] = []
    formated_inputs[ik]['l1charge'] = []
    formated_inputs[ik]['l1pfx'] = []
    formated_inputs[ik]['l1sip'] = []
    formated_inputs[ik]['l1pdg'] = []
    formated_inputs[ik]['l2pt'] = []
    formated_inputs[ik]['l2eta'] = []
    formated_inputs[ik]['l2phi'] = []
    formated_inputs[ik]['l2charge'] = []
    formated_inputs[ik]['l2pfx'] = []
    formated_inputs[ik]['l2sip'] = []
    formated_inputs[ik]['l2pdg'] = []
    formated_inputs[ik]['l3pt'] = []
    formated_inputs[ik]['l3eta'] = []
    formated_inputs[ik]['l3phi'] = []
    formated_inputs[ik]['l3charge'] = []
    formated_inputs[ik]['l3pfx'] = []
    formated_inputs[ik]['l3sip'] = []
    formated_inputs[ik]['l3pdg'] = []
    formated_inputs[ik]['l4pt'] = []
    formated_inputs[ik]['l4eta'] = []
    formated_inputs[ik]['l4phi'] = []
    formated_inputs[ik]['l4charge'] = []
    formated_inputs[ik]['l4pfx'] = []
    formated_inputs[ik]['l4sip'] = []
    formated_inputs[ik]['l4pdg'] = []
    formated_inputs[ik]['isomax'] = []
    formated_inputs[ik]['sipmax'] = []
    formated_inputs[ik]['z1mass'] = []
    formated_inputs[ik]['z2mass'] = []
    formated_inputs[ik]['costhetastar'] = []
    formated_inputs[ik]['costheta1'] = []
    formated_inputs[ik]['costheta2'] = []
    formated_inputs[ik]['thetastar'] = []
    formated_inputs[ik]['theta1'] = []
    formated_inputs[ik]['theta2'] = []
    formated_inputs[ik]['phi'] = []
    formated_inputs[ik]['phistar1'] = []
    formated_inputs[ik]['pt4l'] = []
    formated_inputs[ik]['eta4l'] = []
    formated_inputs[ik]['mass4l'] = []
    formated_inputs[ik]['njets'] = []
    formated_inputs[ik]['deltajj'] = []
    formated_inputs[ik]['massjj'] = []
    formated_inputs[ik]['j1pt'] = []
    formated_inputs[ik]['j1eta'] = []
    formated_inputs[ik]['j1phi'] = []
    formated_inputs[ik]['j1e'] = []
    formated_inputs[ik]['j2pt'] = []
    formated_inputs[ik]['j2eta'] = []
    formated_inputs[ik]['j2phi'] = []
    formated_inputs[ik]['j2e'] = []
    formated_inputs[ik]['j3pt'] = []
    formated_inputs[ik]['j3eta'] = []
    formated_inputs[ik]['j3phi'] = []
    formated_inputs[ik]['j3e'] = []
    formated_inputs[ik]['j4pt'] = []
    formated_inputs[ik]['j4eta'] = []
    formated_inputs[ik]['j4phi'] = []
    formated_inputs[ik]['j4e'] = []
    formated_inputs[ik]['j5pt'] = []
    formated_inputs[ik]['j5eta'] = []
    formated_inputs[ik]['j5phi'] = []
    formated_inputs[ik]['j5e'] = []
    formated_inputs[ik]['j6pt'] = []
    formated_inputs[ik]['j6eta'] = []
    formated_inputs[ik]['j6phi'] = []
    formated_inputs[ik]['j6e'] = []
    formated_inputs[ik]['pfmet'] = []
    formated_inputs[ik]['genmet'] = []
    formated_inputs[ik]['mT'] = []
    formated_inputs[ik]['dphi'] = []
    formated_inputs[ik]['nbjets'] = []
    
    
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
            
        #gets the sum of weights
        event_index[imc] += 1
        mc_sumweight[imc] += evt.f_weight

        formated_inputs[imc]['mc'].append( imc )#0
        if(file_name.find('histos4mu') != -1):
            formated_inputs[imc]['final_state'].append( '4mu' )
        if(file_name.find('histos4e') != -1):
            formated_inputs[imc]['final_state'].append( '4e' )
        if(file_name.find('histos2e2mu') != -1):
            formated_inputs[imc]['final_state'].append( '2e2mu' )
        formated_inputs[imc]['event_index'].append( ievt )#0
        formated_inputs[imc]['mc_sumweight'].append( evt.f_weight )#0 it's replaced in the end by weights sum
        formated_inputs[imc]['event_weight'].append( evt.f_weight )#0
        formated_inputs[imc]['Djet'].append( evt.f_D_jet )#1
        formated_inputs[imc]['DjetVAJHU'].append( evt.f_Djet_VAJHU )#2
        formated_inputs[imc]['l1pt'].append( evt.f_lept1_pt )#3
        formated_inputs[imc]['l1eta'].append( evt.f_lept1_eta )#4
        formated_inputs[imc]['l1phi'].append( evt.f_lept1_phi )#5
        formated_inputs[imc]['l1charge'].append( evt.f_lept1_charge )#6
        formated_inputs[imc]['l1pfx'].append( evt.f_lept1_pfx )#7
        formated_inputs[imc]['l1sip'].append( evt.f_lept1_sip )#8
        formated_inputs[imc]['l1pdg'].append( evt.f_lept1_pdgid )#9
        formated_inputs[imc]['l2pt'].append( evt.f_lept2_pt )#10
        formated_inputs[imc]['l2eta'].append( evt.f_lept2_eta )#11
        formated_inputs[imc]['l2phi'].append( evt.f_lept2_phi )#12
        formated_inputs[imc]['l2charge'].append( evt.f_lept2_charge )#13
        formated_inputs[imc]['l2pfx'].append( evt.f_lept2_pfx )#14
        formated_inputs[imc]['l2sip'].append( evt.f_lept2_sip )#15
        formated_inputs[imc]['l2pdg'].append( evt.f_lept2_pdgid )#16
        formated_inputs[imc]['l3pt'].append( evt.f_lept3_pt )#17
        formated_inputs[imc]['l3eta'].append( evt.f_lept3_eta )#18
        formated_inputs[imc]['l3phi'].append( evt.f_lept3_phi )#19
        formated_inputs[imc]['l3charge'].append( evt.f_lept3_charge )#20
        formated_inputs[imc]['l3pfx'].append( evt.f_lept3_pfx )#21
        formated_inputs[imc]['l3sip'].append( evt.f_lept3_sip )#22
        formated_inputs[imc]['l3pdg'].append( evt.f_lept3_pdgid )#23
        formated_inputs[imc]['l4pt'].append( evt.f_lept4_pt )#24
        formated_inputs[imc]['l4eta'].append( evt.f_lept4_eta )#25
        formated_inputs[imc]['l4phi'].append( evt.f_lept4_phi )#26
        formated_inputs[imc]['l4charge'].append( evt.f_lept4_charge )#27
        formated_inputs[imc]['l4pfx'].append( evt.f_lept4_pfx )#28
        formated_inputs[imc]['l4sip'].append( evt.f_lept4_sip )#29
        formated_inputs[imc]['l4pdg'].append( evt.f_lept4_pdgid )#30
        formated_inputs[imc]['isomax'].append( evt.f_iso_max )#31
        formated_inputs[imc]['sipmax'].append( evt.f_sip_max )#32
        formated_inputs[imc]['z1mass'].append( evt.f_Z1mass )#33
        formated_inputs[imc]['z2mass'].append( evt.f_Z2mass )#34
        formated_inputs[imc]['costhetastar'].append( evt.f_angle_costhetastar )#35
        formated_inputs[imc]['costheta1'].append( evt.f_angle_costheta1 )#36
        formated_inputs[imc]['costheta2'].append( evt.f_angle_costheta2 )#37
        formated_inputs[imc]['thetastar'].append( math.acos(evt.f_angle_costhetastar) )#35
        formated_inputs[imc]['theta1'].append( math.acos(evt.f_angle_costheta1) )#36
        formated_inputs[imc]['theta2'].append( math.acos(evt.f_angle_costheta2) )#37
        formated_inputs[imc]['phi'].append( evt.f_angle_phi )#38
        formated_inputs[imc]['phistar1'].append( evt.f_angle_phistar1 )#39
        formated_inputs[imc]['pt4l'].append( evt.f_pt4l )#40
        formated_inputs[imc]['eta4l'].append( evt.f_eta4l )#41
        formated_inputs[imc]['mass4l'].append( evt.f_mass4l )#42
        formated_inputs[imc]['njets'].append( evt.f_njets_pass )#43
        formated_inputs[imc]['deltajj'].append( evt.f_deltajj )#44
        formated_inputs[imc]['massjj'].append( evt.f_massjj )#45
        formated_inputs[imc]['j1pt'].append( evt.f_jet1_highpt_pt )#46
        formated_inputs[imc]['j1eta'].append( evt.f_jet1_highpt_eta )#47
        formated_inputs[imc]['j1phi'].append( evt.f_jet1_highpt_phi )#48
        formated_inputs[imc]['j1e'].append( evt.f_jet1_highpt_e*math.cosh(evt.f_jet1_highpt_eta) )#49
        formated_inputs[imc]['j2pt'].append( evt.f_jet2_highpt_pt )#50
        formated_inputs[imc]['j2eta'].append( evt.f_jet2_highpt_eta )#51
        formated_inputs[imc]['j2phi'].append( evt.f_jet2_highpt_phi )#52
        formated_inputs[imc]['j2e'].append( evt.f_jet2_highpt_e*math.cosh(evt.f_jet2_highpt_eta) )#53
        
        #3rd highest pt jet
        if( evt.f_jet3_highpt_pt == -999 ):#set inputs to 0 if not initialized
            formated_inputs[imc]['j3pt'].append( default_value )#54
            formated_inputs[imc]['j3eta'].append( default_value )#55
            formated_inputs[imc]['j3phi'].append( default_value )#56
            formated_inputs[imc]['j3e'].append( default_value )#57
        else:
            formated_inputs[imc]['j3pt'].append( evt.f_jet3_highpt_pt )#54
            formated_inputs[imc]['j3eta'].append( evt.f_jet3_highpt_eta )#55
            formated_inputs[imc]['j3phi'].append( evt.f_jet3_highpt_phi )#56
            formated_inputs[imc]['j3e'].append( evt.f_jet3_highpt_e*math.cosh(evt.f_jet3_highpt_eta) )#57

	#4rd highest pt jet
        if( evt.f_jet3_highpt_pt == -999 ):#set inputs to 0 if not initialized
            formated_inputs[imc]['j4pt'].append( default_value )#54
            formated_inputs[imc]['j4eta'].append( default_value )#55
            formated_inputs[imc]['j4phi'].append( default_value )#56
            formated_inputs[imc]['j4e'].append( default_value )#57
        else:
            formated_inputs[imc]['j4pt'].append( evt.f_jet3_highpt_pt )#54
            formated_inputs[imc]['j4eta'].append( evt.f_jet3_highpt_eta )#55
            formated_inputs[imc]['j4phi'].append( evt.f_jet3_highpt_phi )#56
            formated_inputs[imc]['j4e'].append( evt.f_jet3_highpt_e*math.cosh(evt.f_jet3_highpt_eta) )#57

	#5rd highest pt jet
        if( evt.f_jet3_highpt_pt == -999 ):#set inputs to 0 if not initialized
            formated_inputs[imc]['j5pt'].append( default_value )#54
            formated_inputs[imc]['j5eta'].append( default_value )#55
            formated_inputs[imc]['j5phi'].append( default_value )#56
            formated_inputs[imc]['j5e'].append( default_value )#57
        else:
            formated_inputs[imc]['j5pt'].append( evt.f_jet3_highpt_pt )#54
            formated_inputs[imc]['j5eta'].append( evt.f_jet3_highpt_eta )#55
            formated_inputs[imc]['j5phi'].append( evt.f_jet3_highpt_phi )#56
            formated_inputs[imc]['j5e'].append( evt.f_jet3_highpt_e*math.cosh(evt.f_jet3_highpt_eta) )#57

	#6rd highest pt jet
        if( evt.f_jet3_highpt_pt == -999 ):#set inputs to 0 if not initialized
            formated_inputs[imc]['j6pt'].append( default_value )#54
            formated_inputs[imc]['j6eta'].append( default_value )#55
            formated_inputs[imc]['j6phi'].append( default_value )#56
            formated_inputs[imc]['j6e'].append( default_value )#57
        else:
            formated_inputs[imc]['j6pt'].append( evt.f_jet3_highpt_pt )#54
            formated_inputs[imc]['j6eta'].append( evt.f_jet3_highpt_eta )#55
            formated_inputs[imc]['j6phi'].append( evt.f_jet3_highpt_phi )#56
            formated_inputs[imc]['j6e'].append( evt.f_jet3_highpt_e*math.cosh(evt.f_jet3_highpt_eta) )#57

        formated_inputs[imc]['pfmet'].append( evt.f_pfmet )#58
        formated_inputs[imc]['genmet'].append( evt.f_genmet )#59
        formated_inputs[imc]['mT'].append( evt.f_mT )#60
        formated_inputs[imc]['dphi'].append( evt.f_dphi )#61
        formated_inputs[imc]['nbjets'].append( evt.f_Nbjets )#62
                              

    tfile.Close()
    #print ('%i - Processed (%i)' % (ifile, nacepted)), file_name


#for randomizing FS composition in samples
print ''
print 'Shuffling events...'
import random
for ik in formated_inputs:
    print ik," -- nevents = ",event_index[ik]," -- sumweight = ",mc_sumweight[ik]
    for ivar in formated_inputs[ik]:
        random.seed(999) #keeping the same seed -> events ordering is preserved
        random.shuffle(formated_inputs[ik][ivar])
        random.seed(348) #keeping the same seed -> events ordering is preserved
        random.shuffle(formated_inputs[ik][ivar])
        
    for iev in range(len(formated_inputs[ik]['mc_sumweight'])):
        formated_inputs[ik]['mc_sumweight'][iev] = mc_sumweight[ik]
    
#save the dictionary
fileout = open('hzz4l_vbf_selection_noDjet_m4l118-130GeV_shuffledFS.pkl','w')
pickle.dump( formated_inputs, fileout )
fileout.close()
