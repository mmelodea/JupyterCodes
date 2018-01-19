#set adequate flag for Theano on lxplus
import theano
theano.config.gcc.cxxflags = '-march=corei7'

#load needed things
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp
import ROOT
import itertools
import math

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#boost particles towards Higgs referencial
def boostToHiggs(p4Origin):
    HiggsBoostVector = p4Origin[0].BoostVector()    
    p4Boosted = []
    for ip in range(len(p4Origin)):
        p4O = p4Origin[ip]
        p4O.Boost(-HiggsBoostVector)
        p4Boosted.append( p4O )
    
    return p4Boosted


#format the inputs from TTree
def formatInputs(files):    
    formated_inputs = []
    sigcode = 11
    for ifile in range(len(files)):
        tfile = ROOT.TFile.Open(files[ifile])
        tree = tfile.Get('HZZ4LeptonsAnalysisReduced')
        
        nacepted = 0
        for ievt, evt in enumerate(tree):
            if(evt.f_njets_pass >= 2 and evt.f_mass4l >= 118 and evt.f_mass4l <= 130): #VBF region
                event = []
            
                if(ifile == sigcode):
                    event.append(1)
                        
                elif(ifile == 9):
                    event.append(0)

                #else:
                #    event.append(0)
                else:
                    break
            
                nacepted += 1

                lep = []
                lep.append( evt.f_lept1_pt )
                lep.append( evt.f_lept1_eta )
                lep.append( evt.f_lept1_phi )
                lep.append( evt.f_lept1_pdgid )
                #lep.append( evt.f_lept1_charge )
                event.append(lep)

                lep = []
                lep.append( evt.f_lept2_pt )
                lep.append( evt.f_lept2_eta )
                lep.append( evt.f_lept2_phi )
                lep.append( evt.f_lept2_pdgid )
                #lep.append( evt.f_lept2_charge )
                event.append(lep)

                lep = []
                lep.append( evt.f_lept3_pt )
                lep.append( evt.f_lept3_eta )
                lep.append( evt.f_lept3_phi )
                lep.append( evt.f_lept3_pdgid )
                #lep.append( evt.f_lept3_charge )
                event.append(lep)
    
                lep = []
                lep.append( evt.f_lept4_pt )
                lep.append( evt.f_lept4_eta )
                lep.append( evt.f_lept4_phi )
                lep.append( evt.f_lept4_pdgid )
                #lep.append( evt.f_lept4_charge )
                event.append(lep)
            
                jet = []
                jet.append( evt.f_jet1_highpt_pt )
                jet.append( evt.f_jet1_highpt_eta )
                jet.append( evt.f_jet1_highpt_phi )
                #jet.append( evt.f_jet1_highpt_e*(ROOT.TMath.CosH(evt.f_jet1_highpt_eta)) )
                event.append(jet)

                jet = []
                jet.append( evt.f_jet2_highpt_pt )
                jet.append( evt.f_jet2_highpt_eta )
                jet.append( evt.f_jet2_highpt_phi )
                #jet.append( evt.f_jet2_highpt_e*(ROOT.TMath.CosH(evt.f_jet2_highpt_eta)) )
                event.append(jet)

                jet = []
                jet.append( evt.f_jet3_highpt_pt )
                jet.append( evt.f_jet3_highpt_eta )
                jet.append( evt.f_jet3_highpt_phi )
                #jet.append( evt.f_jet3_highpt_e*(ROOT.TMath.CosH(evt.f_jet3_highpt_eta)) )
                event.append(jet)
                
                event.append(evt.f_D_jet) #CMS VBF discriminant
                event.append(evt.f_Djet_VAJHU) #MELA
                event.append(evt.f_weight) #event weight
                event.append(evt.f_run)
                event.append(evt.f_lumi)
                event.append(evt.f_event)
                
                formated_inputs.append(event)
                if(nacepted > 200):
                    break

        print ('Processed (%i)' % nacepted), tfile.GetName()
        if(ifile == sigcode):
            print '>>> SIGNAL FILE <<<'

    return formated_inputs


#loads input data
file_names2e2mu = open('/afs/cern.ch/work/m/mmelodea/private/Higgs_Ntuples/files2016/histos2e2mu_25ns/filelist_2e2mu_2016_Spring16_AN_Bari_MC.txt','r')
file_names4e = open('/afs/cern.ch/work/m/mmelodea/private/Higgs_Ntuples/files2016/histos4e_25ns/filelist_4e_2016_Spring16_AN_Bari_MC.txt','r')
file_names4mu = open('/afs/cern.ch/work/m/mmelodea/private/Higgs_Ntuples/files2016/histos4mu_25ns/filelist_4mu_2016_Spring16_AN_Bari_MC.txt','r')
path = '/afs/cern.ch/work/m/mmelodea/private/Higgs_Ntuples/files2016/'
files2e2mu = [path+'histos2e2mu_25ns/'+i.rstrip() for i in file_names2e2mu.readlines()]
files4e = [path+'histos4e_25ns/'+i.rstrip() for i in file_names4e.readlines()]
files4mu = [path+'histos4mu_25ns/'+i.rstrip() for i in file_names4mu.readlines()]

events2e2mu = formatInputs(files2e2mu)
events4e = formatInputs(files4e)
events4mu = formatInputs(files4mu)

print 'events2e2mu: %i' % len(events2e2mu)
print 'events4e: %i' % len(events4e)
print 'events4mu: %i' % len(events4mu)


def include_subjets(events, subjets_files):
    nevents = len(events)
    for ie in range(nevents):
        print 'Remaining %i' % (nevents-ie)
        run = events[ie][11]
        event = events[ie][12]
        lumi = events[ie][13]
        
        for i in range(len(subjets_files)):        
            tfile = ROOT.TFile.Open(subjets_files[i])
            tree = tfile.Get('JetImage')
            for ievt, evt in enumerate(tree):
                Run = evt.Run
                Event = evt.Event
                Lumi = evt.Lumi
                        
            if(Run == run and Event == event and Lumi == lumi):
                subjets = []
                for sbj in range(len(evt.SubJetPt)):
                    eta = [ieta for ieta in evt.SubJetEta[sbj]]
                    phi = [iphi for iphi in evt.SubJetPhi[sbj]]
                    pt = [ipt for ipt in evt.SubJetPt[sbj]]
                subjets.append( [eta, phi, pt] )
                events[ie].append( subjets )
                break
                            
    return events


#subjets
file_vbf = '/afs/cern.ch/work/m/mmelodea/private/MonoHiggs/CMSSW_9_0_0/src/JetImageFiles/VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root'
file_ggh = '/afs/cern.ch/work/m/mmelodea/private/MonoHiggs/CMSSW_9_0_0/src/JetImageFiles/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root'
subjets_files = [file_vbf, file_ggh]

events = []
for iv in events2e2mu:
    events.append(iv)
for iv in events4e:
    events.append(iv)
for iv in events4mu:
    events.append(iv)

events = include_subjets(events, subjets_files)
print events[0]


#reconstruct Z's and Higgs and organize things
events = [events2e2mu, events4e, events4mu]

full_event = {}
full_event['signal'] = []
full_event['background'] = []

max_per_ch = 1000000
for ich in range(len(events)):
    snch = 0
    bnch = 0
    for iev in range(len(events[ich])):
        lp4 = [ROOT.TLorentzVector() for i in range(4)]
        for il in range(4):
            mass = 0.
            if(abs(events[ich][iev][il+1][3]) == 13):
                mass = 0.106
            lp4[il].SetPtEtaPhiM(events[ich][iev][il+1][0],events[ich][iev][il+1][1],events[ich][iev][il+1][2],mass)
        
        z1 = lp4[0] + lp4[1]
        z2 = lp4[2] + lp4[3]
        h = z1 + z2
    
        if(events[ich][iev][0] == 1):
            snch += 1
            if(snch > max_per_ch):
                continue
            
            full_event['signal'].append([
                [h.Pt(),h.Eta(),h.Phi(),h.E()],                      #Higgs p4
                [z1.Pt(),z1.Eta(),z1.Phi(),z1.E()],                  #Z1 p4
                [z2.Pt(),z2.Eta(),z2.Phi(),z2.E()],                  #Z2 p4
                [lp4[0].Pt(),lp4[0].Eta(),lp4[0].Phi(),lp4[0].E()],  #l1 p4
                [lp4[1].Pt(),lp4[1].Eta(),lp4[1].Phi(),lp4[1].E()],  #l2 p4
                [lp4[2].Pt(),lp4[2].Eta(),lp4[2].Phi(),lp4[2].E()],  #l3 p4
                [lp4[3].Pt(),lp4[3].Eta(),lp4[3].Phi(),lp4[3].E()],  #l4 p4
                events[ich][iev][5],                                 #j1 p4
                events[ich][iev][6],                                 #j2 p4
                events[ich][iev][7],                                 #j3 p4
                events[ich][iev][8],                                 #Djet
                events[ich][iev][9]                                  #MELA
            ])
        else:
            bnch += 1
            if(bnch > max_per_ch):
                continue
            
            full_event['background'].append([
                [h.Pt(),h.Eta(),h.Phi(),h.E()],                      #Higgs p4
                [z1.Pt(),z1.Eta(),z1.Phi(),z1.E()],                  #Z1 p4
                [z2.Pt(),z2.Eta(),z2.Phi(),z2.E()],                  #Z2 p4
                [lp4[0].Pt(),lp4[0].Eta(),lp4[0].Phi(),lp4[0].E()],  #l1 p4
                [lp4[1].Pt(),lp4[1].Eta(),lp4[1].Phi(),lp4[1].E()],  #l2 p4
                [lp4[2].Pt(),lp4[2].Eta(),lp4[2].Phi(),lp4[2].E()],  #l3 p4
                [lp4[3].Pt(),lp4[3].Eta(),lp4[3].Phi(),lp4[3].E()],  #l4 p4
                events[ich][iev][5],                                 #j1 p4
                events[ich][iev][6],                                 #j2 p4
                events[ich][iev][7],                                 #j3 p4
                events[ich][iev][8],                                 #Djet
                events[ich][iev][9]                                  #MELA
            ])
    
    
print '# Sig Events: %i' % len(full_event['signal'])
print '# Bkg Events: %i' % len(full_event['background'])
#print full_event['signal'][0]



# Run classifier with cross-validation and plot ROC curves
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

#esY = np.ones(len(full_event['signal']))
#ebY = np.zeros(len(full_event['background']))
#Yexp = np.concatenate([esY, ebY])
#encoder = LabelEncoder()
#encoder.fit(Yexp)
#encoded_Y = encoder.transform(Yexp)
#print encoded_Y

#osY = [full_event['signal'][i][8] for i in range(len(full_event['signal']))]
#obY = [full_event['background'][i][8] for i in range(len(full_event['background']))]
#Yobs = np.concatenate([osY,obY])
#fpr, tpr, thresholds = roc_curve(encoded_Y, Yobs)
#roc_auc = auc(fpr, tpr)
#print '----->> Djet ROC area (all events): %.2f' % roc_auc

#osY = [full_event['signal'][i][9] for i in range(len(full_event['signal']))]
#obY = [full_event['background'][i][9] for i in range(len(full_event['background']))]
#Yobs = np.concatenate([osY,obY])
#fpr, tpr, thresholds = roc_curve(encoded_Y, Yobs)
#roc_auc = auc(fpr, tpr)
#print '----->> MELA ROC area (all events): %.2f' % roc_auc




#shuffle the channels
np.random.shuffle(full_event['signal'])
np.random.shuffle(full_event['background'])



from keras.layers import LSTM, Bidirectional, Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam


# Bidirectional LSTM Model
def build_bilstm_model(n_cand_per_jet, features_per_jet):
    # Headline input: meant to receive sequences of 200 floats
    # Note that we can name any layer by passing it a "name" argument.
    i = Input(shape=(n_cand_per_jet, features_per_jet,), name='main_input')
    # the masking layer will prevent the LSTM layer to consider the 0-padded jet values
    m = Masking()(i)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    # the Bidirectional will make the LSTM cell read the sequence from end to start and start to end at the same time
    m = Bidirectional( LSTM(50) ) (m)

    
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(m)
    model = Model(input=[i], output=[auxiliary_output])
    #opt = SGD()
    opt = Adam()
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model



njets = 61000
n_cand_per_jet = 10
features = {
    'pt' : None,
    'eta': None
    #'phi': None,
    #'e'  : None
}
features_per_jet = len(features)
model = build_bilstm_model(n_cand_per_jet, features_per_jet)
print model.summary()



momentum_input = {}
momentum_input['signal'] = np.zeros((njets, n_cand_per_jet, features_per_jet)) # 3 momentum
momentum_input['background'] = np.zeros((njets, n_cand_per_jet, features_per_jet)) # 3 momentum


for i in range(njets):
    for j in range(n_cand_per_jet):
        for iprop in range(features_per_jet):
            if(full_event['signal'][i][j][iprop] != -999.):
                momentum_input['signal'][i][j][iprop] = full_event['signal'][i][j][iprop]
            else:
                break
        for iprop in range(features_per_jet):
            if(full_event['background'][i][j][iprop] != -999.):
                momentum_input['background'][i][j][iprop] = full_event['background'][i][j][iprop]
            else:
                break
            
            

#train the network
X = np.concatenate([momentum_input['signal'], momentum_input['background']])
Y_TT = np.ones(momentum_input['signal'].shape[0])
Y_QCD = np.zeros(momentum_input['background'].shape[0])
Y = np.concatenate([Y_TT, Y_QCD])

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

kfold = StratifiedKFold(n_splits=2, shuffle=True,  random_state=seed)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'black', 'green', 'brown'])
lw = 2

i = 0
histories = []
for i,((train, test), color) in enumerate(zip(kfold.split(X, encoded_Y), colors)):
    print "\t\tFold",i
    bilstm_model = build_bilstm_model(n_cand_per_jet, features_per_jet)
    history = bilstm_model.fit(X[train], encoded_Y[train], 
                               validation_data=(X[test], encoded_Y[test]), 
                               epochs=100, batch_size=128, 
                               verbose=2, callbacks=[early_stopping])
    Y_score = bilstm_model.predict(X[test])
    histories.append(history)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(encoded_Y[test], Y_score)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pyp.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

pyp.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
mean_tpr /= kfold.get_n_splits(X, encoded_Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'ROC AUC: %.2f' % mean_auc

#pyp.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
#pyp.xlim([0, 1.0])
#pyp.ylim([0, 1.0])
#pyp.xlabel('False Positive Rate')
#pyp.ylabel('True Positive Rate')
#pyp.title('Receiver operating characteristic example')
#pyp.legend(loc="lower right")
#pyp.show()






#over all events
momentum_input['signal'] = np.zeros((len(full_event['signal']), n_cand_per_jet, features_per_jet)) # 3 momentum
momentum_input['background'] = np.zeros((len(full_event['background']), n_cand_per_jet, features_per_jet)) # 3 momentum

for i in range(len(full_event['signal'])):
    for j in range(n_cand_per_jet):
        for iprop in range(features_per_jet):
            if(full_event['signal'][i][j][iprop] != -999.):
                momentum_input['signal'][i][j][iprop] = full_event['signal'][i][j][iprop]
            else:
                break
                
for i in range(len(full_event['background'])):
    for j in range(n_cand_per_jet):
        for iprop in range(features_per_jet):
            if(full_event['background'][i][j][iprop] != -999.):
                momentum_input['background'][i][j][iprop] = full_event['background'][i][j][iprop]
            else:
                break

X = np.concatenate([momentum_input['signal'], momentum_input['background']])
Y_TT = np.ones(momentum_input['signal'].shape[0])
Y_QCD = np.zeros(momentum_input['background'].shape[0])
Y = np.concatenate([Y_TT, Y_QCD])
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

Y_score = bilstm_model.predict(X)
fpr, tpr, thresholds = roc_curve(encoded_Y, Y_score)
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %.2f' % roc_auc
