#defines the comparator for #jets
def comparator(a, b, comparison):
    if(comparison == '==' and a == b):
        return True
    elif(comparison == '>=' and a >= b):
        return True
    elif(comparison == '>' and a > b):
        return True
    else:
        return False
    
    
    
#format inputs from TTree (Data & MC)
def formatInputs(files, comparison, njets, min_m4l, max_m4l):    
    import ROOT
    
    formated_inputs = {}
    keys = ['Data','VBF','HJJ','ttH','ZH','WH','ggZZ','qqZZ','ttZ','ggH','myqqZZ']
    for ik in keys:
        formated_inputs[ik] = []

    for ifile in range(len(files)):
        file_name = files[ifile]
        
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

        print 'Processing %s' % file_name
        tfile = ROOT.TFile.Open(file_name)
        tree = tfile.Get('HZZ4LeptonsAnalysisReduced')
        
        for ievt, evt in enumerate(tree):
            #VBF region
            if(comparator(evt.f_njets_pass, njets, comparison) and evt.f_mass4l >= min_m4l and evt.f_mass4l <= max_m4l):                 
                
                event = []        
                event.append(evt.f_D_jet) #CMS VBF discriminant
                event.append(evt.f_Djet_VAJHU) #MELA
                event.append(evt.f_weight) #event weight
                
                event.append(evt.f_mass4l)
                event.append(evt.f_Z1mass)
                event.append(evt.f_Z2mass)                

                lep = []
                lep.append( evt.f_lept1_pt )
                lep.append( evt.f_lept1_eta )
                lep.append( evt.f_lept1_phi )
                lep.append( evt.f_lept1_pdgid )
                event.append(lep)

                lep = []
                lep.append( evt.f_lept2_pt )
                lep.append( evt.f_lept2_eta )
                lep.append( evt.f_lept2_phi )
                lep.append( evt.f_lept2_pdgid )
                event.append(lep)

                lep = []
                lep.append( evt.f_lept3_pt )
                lep.append( evt.f_lept3_eta )
                lep.append( evt.f_lept3_phi )
                lep.append( evt.f_lept3_pdgid )
                event.append(lep)
    
                lep = []
                lep.append( evt.f_lept4_pt )
                lep.append( evt.f_lept4_eta )
                lep.append( evt.f_lept4_phi )
                lep.append( evt.f_lept4_pdgid )
                event.append(lep)
            
                for ijet in range(6):
                    jet = []
                    jet.append(evt.f_jets_dnn_pt[ijet])
                    jet.append(evt.f_jets_dnn_eta[ijet])
                    jet.append(evt.f_jets_dnn_phi[ijet])
                    jet.append(evt.f_jets_dnn_e[ijet]*ROOT.TMath.CosH(evt.f_jets_dnn_eta[ijet]))
                    event.append(jet)
                    
                event.append( evt.f_angle_costhetastar )
                event.append( evt.f_angle_costheta1 )
                event.append( evt.f_angle_costheta2 )
                event.append( evt.f_angle_phi )
                event.append( evt.f_angle_phistar1 )
                                                
                formated_inputs[imc].append(event)
                        

        tfile.Close()
        #print ('%i - Processed (%i)' % (ifile, nacepted)), file_name

    return formated_inputs




#organize events and splits data in train and test set
def splitInputs(events, train_fraction, class_weight):
    import ROOT
    import random
    
    full_event_train = []
    full_event_test = []
    full_event_data = []    
        
    for ik in events:
        if(ik == 'Data'):
                for iev in range(len(events[ik])):
                    full_event_data.append([
                        ik,                  #MC ID
                        events[ik][iev][0],  #Djet
                        events[ik][iev][1],  #MELA
                        events[ik][iev][2],  #event weight
                        class_weight[ik],    #events sum weight
                        events[ik][iev][3], #Higgs mass
                        events[ik][iev][4], #Z1 mass
                        events[ik][iev][5],  #Z2 mass
                        events[ik][iev][6],  #l1
                        events[ik][iev][7],  #l2
                        events[ik][iev][8],  #l3
                        events[ik][iev][9],  #l4
                        events[ik][iev][10],  #j1
                        events[ik][iev][11],  #j2
                        events[ik][iev][12],  #j3
                        events[ik][iev][13],  #j4
                        events[ik][iev][14],  #j5
                        events[ik][iev][15],  #j6
                        events[ik][iev][16],  #costhetastar
                        events[ik][iev][17],  #costheta1
                        events[ik][iev][18],  #costheta2
                        events[ik][iev][19],  #phi
                        events[ik][iev][20]   #phistar1
                    ])

        else:
                for iev in range(len(events[ik])):
                    if(iev < int(len(events[ik])*train_fraction)):
                        full_event_train.append([
                            ik,                  #MC ID
                            events[ik][iev][0],  #Djet
                            events[ik][iev][1],  #MELA
                            events[ik][iev][2],  #event weight                
                            class_weight[ik],    #events sum weight
                            events[ik][iev][3], #Higgs mass
                            events[ik][iev][4], #Z1 mass
                            events[ik][iev][5],  #Z2 mass
                            events[ik][iev][6],  #l1
                            events[ik][iev][7],  #l2
                            events[ik][iev][8],  #l3
                            events[ik][iev][9],  #l4
                            events[ik][iev][10],  #j1
                            events[ik][iev][11],  #j2
                            events[ik][iev][12],  #j3
                            events[ik][iev][13],  #j4
                            events[ik][iev][14],  #j5
                            events[ik][iev][15],  #j6
                            events[ik][iev][16],  #costhetastar
                            events[ik][iev][17],  #costheta1
                            events[ik][iev][18],  #costheta2
                            events[ik][iev][19],  #phi
                            events[ik][iev][20]   #phistar1                            
                        ])
                        
                    else:
                        #my private qqZZ sample it's not used in testing
                        if(ik == 'myqqZZ'):
                            continue
                        full_event_test.append([
                            ik,                  #MC ID
                            events[ik][iev][0],  #Djet
                            events[ik][iev][1],  #MELA
                            events[ik][iev][2],  #event weight                
                            class_weight[ik],    #events sum weight
                            events[ik][iev][3], #Higgs mass
                            events[ik][iev][4], #Z1 mass
                            events[ik][iev][5],  #Z2 mass
                            events[ik][iev][6],  #l1
                            events[ik][iev][7],  #l2
                            events[ik][iev][8],  #l3
                            events[ik][iev][9],  #l4
                            events[ik][iev][10],  #j1
                            events[ik][iev][11],  #j2
                            events[ik][iev][12],  #j3
                            events[ik][iev][13],  #j4
                            events[ik][iev][14],  #j5
                            events[ik][iev][15],  #j6
                            events[ik][iev][16],  #costhetastar
                            events[ik][iev][17],  #costheta1
                            events[ik][iev][18],  #costheta2
                            events[ik][iev][19],  #phi
                            events[ik][iev][20]   #phistar1                            
                        ])
            
    print 'Train set size: %i' % len(full_event_train)
    print 'Test set size: %i' % len(full_event_test)
    print 'Data set size: %i' % len(full_event_data)
    #shuffles events
    random.seed(6)
    random.shuffle(full_event_train)
    random.seed(93)
    random.shuffle(full_event_test)
    
    return full_event_train, full_event_test, full_event_data



#prepare sets for training
def prepareSet(event_set, nparticles, nfeatures, djet_index, mela_index, weight_index, class_weight_index):
    import numpy as np
    from math import cos, sin, cosh, sinh, fabs, pow, sqrt
    
    nevents = len(event_set)
    inputs = np.zeros((nevents, nparticles*nfeatures))    
    labels = []
    weights = []        
    scales = []    
    djet = []
    mela = []
    
    siev = 0
    biev = 0
    for iev in range(nevents):
        djet.append( event_set[iev][djet_index] )
        mela.append( event_set[iev][mela_index] )
        scales.append( event_set[iev][class_weight_index])
        weights.append( event_set[iev][weight_index] )
        for j in range(nparticles):
            for iprop in range(nfeatures):
                if(event_set[iev][j+8][iprop] != -999.):
                    #for pt, eta, phi
                    inputs[iev][j*nfeatures+iprop] = event_set[iev][j+8][iprop]
                    
                    #for px, py, pz
                    #px = event_set[iev][j+8][0]*cos(event_set[iev][j+8][2])
                    #py = event_set[iev][j+8][0]*sin(event_set[iev][j+8][2])
                    #pz = event_set[iev][j+8][0]*sinh(event_set[iev][j+8][1])
                    #if(iprop == 0):#px
                    #    inputs[iev][j*nfeatures+iprop] = px
                    #if(iprop == 1):#py
                    #    inputs[iev][j*nfeatures+iprop] = py
                    #if(iprop == 2):#pz
                    #    inputs[iev][j*nfeatures+iprop] = pz
                    
                    #for e
                    #if(iprop == 3):#energy
                    #    if(j < 4 and fabs(event_set[iev][j+8][3]) == 11):
                    #        inputs[iev][j*nfeatures+iprop] = sqrt(pow(0.000511,2) + pow(px,2) + pow(py,2) + pow(pz,2))
                    #    if(j < 4 and fabs(event_set[iev][j+8][3]) == 13):
                    #        inputs[iev][j*nfeatures+iprop] = sqrt(pow(0.105658,2) + pow(px,2) + pow(py,2) + pow(pz,2))
                    #    if(j >= 4):#jetE = eT*cosH(eta)
                    #        inputs[iev][j*nfeatures+iprop] = event_set[iev][j+8][3]*cosh(event_set[iev][j+8][1])

        if(event_set[iev][0] == 'VBF'):
            siev += 1
            labels.append(1)

        if(event_set[iev][0] != 'VBF' and event_set[iev][0] != 'Data'):
            biev += 1
            labels.append(0)
        
        #if(event_set[iev][0] == 'VBF'):
        #    labels.append(0)
        #if(event_set[iev][0] == 'ttH' or event_set[iev][0] == 'WH' or event_set[iev][0] == 'ZH'):
        #    labels.append(1)
        #if(event_set[iev][0] == 'HJJ'):
        #    labels.append(2)
        #if(event_set[iev][0] == 'qqZZ' or event_set[iev][0] == 'myqqZZ' or event_set[iev][0] == 'ggZZ'):
        #    labels.append(3)
        #if(event_set[iev][0] == 'ttZ'):
        #    labels.append(4)
        
    print 'siev: %i' % siev
    print 'biev: %i' % biev
    ainputs = np.asarray(inputs)
    alabels = np.asarray(labels)
    aweights = np.asarray(weights)
    ascales = np.asarray(scales)
    return ainputs, alabels, djet, mela, aweights, ascales




#boost particles
def boostToHiggs(lepsp4, higgsp4):
    HiggsBoostVector = higgsp4.BoostVector()    
    lepsp4_boosted = []
    for il in range(len(lepsp4)):
        p4lep = lepsp4[il]
        if(p4lep.Pt() != -999):
            p4lep.Boost(-HiggsBoostVector)
        lepsp4_boosted.append( p4lep )
    
    return lepsp4_boosted



#Data Augmentation function
def data_augmentation(events_to_da, rsf, gsigma):#rsf = replication scale factor
    from ROOT import TRandom3
    import math
    
    limit = len(events_to_da)
    print 'Event before DA - total: {0}'.format(limit)
    print events_to_da[0]

    #replicate events with perturbation (data augmentation)
    rd = TRandom3()
    da_events = []
    for sf in range(rsf):
        if(sf == 0):
            for iev in range(len(events_to_da)):
                da_events.append( events_to_da[iev] )
        else:
            for iev in range(len(events_to_da)):
                evt = []
                for ip in range(len(events_to_da[iev])):
                    if(ip < 6):
                        evt.append( events_to_da[iev][ip] )
                    else:
                        prop = []
                        for ipp in range(len(events_to_da[iev][ip])):
                            if(ipp == 2 and events_to_da[iev][ip][ipp] != -999):
                                phi = 4
                                while(math.fabs(phi) > 3.14):
                                    phi = events_to_da[iev][ip][ipp]*rd.Gaus(1,gsigma)
                                prop.append( phi )
                            #if(events_to_da[iev][ip][ipp] != -999):
                            #    prop.append( events_to_da[iev][ip][ipp]*rd.Gaus(1,gsigma) )
                            else:
                                prop.append( events_to_da[iev][ip][ipp] )
                        evt.append(prop)
                da_events.append(evt)
    
    print '\nEvent after DA - total: {0}'.format(len(da_events))
    print da_events[limit]

    return da_events