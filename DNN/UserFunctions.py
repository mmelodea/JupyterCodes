#organize events and splits data in train, test and real data sets
def splitInputs(events, train_fraction):
    import random
    
    full_event_train = {}
    full_event_test = {}
    full_event_data = {}    
    ntrain = 0
    ntest = 0
    ndata = 0
        
    for ik in events:
        if(ik == 'Data'):
            full_event_data = events[ik]
            ndata = len(events[ik]['mc'])
                    
        else:
            full_event_train[ik] = {}
            full_event_test[ik] = {}
            nevents = len(events[ik]['mc'])
            for ivar in events[ik]:
                full_event_train[ik][ivar] = []
                full_event_test[ik][ivar] = []
                for iev in range(len(events[ik][ivar])):
                    if(iev < int(nevents*train_fraction)):
                        full_event_train[ik][ivar].append( events[ik][ivar][iev] )
                        if(ivar == 'mc'): #to count just once
                            ntrain += 1
                    else:
                        full_event_test[ik][ivar].append( events[ik][ivar][iev] )
                        if(ivar == 'mc'): #to count just once
                            ntest += 1
            
    print 'Train set size: %i' % ntrain
    print 'Test set size: %i' % ntest
    print 'Data set size: %i' % ndata
    
    return full_event_train, full_event_test, full_event_data



#prepare sets for training
def prepareSet(event_set, use_vars):
    import numpy as np
    from math import cos, sin, cosh, sinh, fabs, pow, sqrt
    
    default_value = 0
    inputs = []
    labels = []
    weights = []        
    scales = []    
    djet = []
    mela = []
    
    siev = 0
    biev = 0
    for ik in event_set:
        for iev in range(len(event_set[ik]['mc'])):
            #stores standard variables
            djet.append( event_set[ik]['Djet'][iev] )
            mela.append( event_set[ik]['DjetVAJHU'][iev] )
            scales.append( event_set[ik]['mc_sumweight'][iev] )
            weights.append( event_set[ik]['event_weight'][iev] )
            
            #defines the event label
            if(ik == 'VBF'):
                siev += 1
                labels.append(1)#signal
            else:
                biev += 1
                labels.append(0)#background

            #defines the vector of inputs to feed network                
            variables = []
            for ivar in use_vars:
                variables.append( event_set[ik][ivar][iev] )
                #sanity check.. make sure no pedestal values are kept
                if( event_set[ik][ivar][iev] == -999 ):
                    variables.append( default_value )
            inputs.append(variables)        
            
                
    print 'siev: %i' % siev
    print 'biev: %i' % biev
    
    #converts to numpy array format (needed for Keras)
    ainputs = np.asarray(inputs)
    alabels = np.asarray(labels)
    aweights = np.asarray(weights)
    ascales = np.asarray(scales)
    return ainputs, alabels, djet, mela, aweights, ascales



#prepare sets for training
def prepareJetComponents(event_set):
    import numpy as np
    
    prepared_set = {}
    scales = {}
    for ik in event_set:
        sum_weight = 0
        prepared_set[ik] = []
        for iev in range(len(event_set[ik])):
            sum_weight += event_set[ik][iev][0]#f_weight
            jets = []
            for ijet in range(3):
                rep = 313*ijet
                jet_eta = []
                jet_phi = []
                jet_pt = []
                jet = []
                if(event_set[ik][iev][43] == 2 and ijet == 2):#f_njets_pass
                    break
                for isjet in range(100):
                    rep2 = 3*isjet
                    #uncomment for jet image in CNN
                    if(isjet > event_set[ik][iev][50+rep]):#subjetness
                        break
                    jet_pt.append( event_set[ik][iev][59+rep+rep2] )#jetcomponentpt
                    jet_eta.append( event_set[ik][iev][60+rep+rep2] )#jetcomponenteta
                    jet_phi.append( event_set[ik][iev][61+rep+rep2] )#jetcomponentphi
                jets.append([jet_eta, jet_phi, jet_pt])
            prepared_set[ik].append(jets)
                    #uncomment for jet classification
                    #jet.append( event_set[ik][iev][59+rep+rep2] )#jetcomponentpt
                    #jet.append( event_set[ik][iev][60+rep+rep2] )#jetcomponenteta
                    #jet.append( event_set[ik][iev][61+rep+rep2] )#jetcomponentphi
                #prepared_set[ik].append(jet)
        scales[ik] = sum_weight
                
    return prepared_set, scales



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
