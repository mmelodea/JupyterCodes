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
                    variables = []
                    variables.append(ik)
                    variables.append( class_weight[ik] )
                    for ivar in range(len(events[ik][iev])):
                        variables.append( events[ik][iev][ivar] )
                    full_event_data.append( variables )
                    
        else:
                for iev in range(len(events[ik])):
                    if(iev < int(len(events[ik])*train_fraction)):
                        variables = []
                        variables.append(ik)
                        variables.append( class_weight[ik] )
                        for ivar in range(len(events[ik][iev])):
                            variables.append( events[ik][iev][ivar] )
                        full_event_train.append( variables )
                        
                    else:
                        #my private qqZZ sample it's not used in testing
                        if(ik == 'myqqZZ'):
                            continue
                        variables = []
                        variables.append(ik)
                        variables.append( class_weight[ik] )
                        for ivar in range(len(events[ik][iev])):
                            variables.append( events[ik][iev][ivar] )
                        full_event_test.append( variables )
            
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
def prepareSet(event_set, djet_index, mela_index, weight_index, class_weight_index):
    import numpy as np
    from math import cos, sin, cosh, sinh, fabs, pow, sqrt
    
    nevents = len(event_set)
    inputs = []
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

        variables = []
        variables.append(event_set[iev][5])#l1pt
        variables.append(event_set[iev][6])#l1eta
        variables.append(event_set[iev][7])#l1phi
        #variables.append(event_set[iev][8])#l1ch
        #variables.append(event_set[iev][9])#l1pfx
        #variables.append(event_set[iev][10])#l1sip
        #variables.append(event_set[iev][11])#l1pdgid
        variables.append(event_set[iev][12])#l2pt
        variables.append(event_set[iev][13])#l2eta
        variables.append(event_set[iev][14])#l2phi
        #variables.append(event_set[iev][15])#l2ch
        #variables.append(event_set[iev][16])#l2pfx
        #variables.append(event_set[iev][17])#l2sip
        #variables.append(event_set[iev][18])#l2pdgid
        variables.append(event_set[iev][19])#l3pt
        variables.append(event_set[iev][20])#l3eta
        variables.append(event_set[iev][21])#l3phi
        #variables.append(event_set[iev][22])#l3ch
        #variables.append(event_set[iev][23])#l3pfx
        #variables.append(event_set[iev][24])#l3sip
        #variables.append(event_set[iev][25])#l3pdgid
        variables.append(event_set[iev][26])#l4pt
        variables.append(event_set[iev][27])#l4eta
        variables.append(event_set[iev][28])#l4phi
        #variables.append(event_set[iev][29])#l4ch
        #variables.append(event_set[iev][30])#l4pfx
        #variables.append(event_set[iev][31])#l4sip
        #variables.append(event_set[iev][32])#l4pdgid
        #variables.append(event_set[iev][33])#isomax
        #variables.append(event_set[iev][34])#sipmax
        #variables.append(event_set[iev][35])#mz1
        #variables.append(event_set[iev][36])#mz2
        #variables.append(event_set[iev][37])#cosO*
        #variables.append(event_set[iev][38])#cosO1
        #variables.append(event_set[iev][39])#cosO2
        #variables.append(event_set[iev][40])#phi
        #variables.append(event_set[iev][41])#phi*_1
        #variables.append(event_set[iev][42])#pt4l
        #variables.append(event_set[iev][43])#eta4l
        #variables.append(event_set[iev][44])#m4l
        #variables.append(event_set[iev][45])#njets
        #variables.append(event_set[iev][46])#detajj
        #variables.append(event_set[iev][47])#mjj
        variables.append(event_set[iev][48])#j1pt
        variables.append(event_set[iev][49])#j1eta
        variables.append(event_set[iev][50])#j1phi
        #variables.append(event_set[iev][51])#j1et
        variables.append(event_set[iev][52])#j2pt
        variables.append(event_set[iev][53])#j2eta
        variables.append(event_set[iev][54])#j2phi
        #variables.append(event_set[iev][55])#j2et

        if(event_set[iev][56] == -999):
            variables.append(0)#j3pt
            variables.append(0)#j3eta
            variables.append(0)#j3phi
            #variables.append(0)#j3et
        else:
            variables.append(event_set[iev][56])#j3pt
            variables.append(event_set[iev][57])#j3eta
            variables.append(event_set[iev][58])#j3phi
            #variables.append(event_set[iev][59])#j3et
            
        #variables.append(event_set[iev][60])#pfmet
        #variables.append(event_set[iev][61])#genmet doesn't exist for data, so we must not use it
        #variables.append(event_set[iev][62])#mt
        #variables.append(event_set[iev][63])#dphi
        #variables.append(event_set[iev][64])#nbjets        
        
        inputs.append(variables)        

        if(event_set[iev][0] == 'VBF'):
            siev += 1
            labels.append(1)

        if(event_set[iev][0] != 'VBF' and event_set[iev][0] != 'Data'):
            biev += 1
            labels.append(0)
            
                
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
