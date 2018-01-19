#set adequate flag for Theano on lxplus
import theano
theano.config.gcc.cxxflags = '-march=corei7'


#check config
import keras.backend as K
K.set_image_dim_ordering('th')


#load needed things
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
from PreProcessingFunctions import prepareImages, translate, rotate, reflect

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#format the inputs from TTree
# 4D tensor (theano backend)
# 1st dim is jet index
# 2nd dim is eta bin
# 3rd dim is phi bin
# 4th dim is props value (pt, charge, pdgId, etc.)
def formatInputs(files):
    quark_jets = []#type 1
    gluon_jets = []#type 0

    for ifile in files:
        tree = ifile.Get('JetImage')
            
        for ievt, evt in enumerate(tree):            
            #get PFJets
            nPFJets = evt.PFJetEta.size()
            for iPFJet in range(nPFJets):
                pfJetDistance = evt.PFJetDistance[iPFJet]
                pfjet = []
                pt  = []
                eta = []
                phi = []
                e = []
                charge = []
                pdgId = []
                    
                #get daughters
                nDaughters = evt.SubJetEta[iPFJet].size()
                for iDau in range(nDaughters):
                    pt.append( evt.SubJetPt[iPFJet][iDau] )
                    eta.append( evt.SubJetEta[iPFJet][iDau] )
                    phi.append( evt.SubJetPhi[iPFJet][iDau] )
                    e.append( evt.SubJetE[iPFJet][iDau] )
                    charge.append( evt.SubJetCharge[iPFJet][iDau] )
                    pdgId.append( evt.SubJetPDGID[iPFJet][iDau] )
                
                #creates the vector of pfjets properties
                pfjet.append(pfJetDistance)
                pfjet.append(pt)
                pfjet.append(eta)
                pfjet.append(phi)
                pfjet.append(e)
                pfjet.append(charge)
                pfjet.append(pdgId)
                
                #decides if quark or gluon jet
                if evt.PFJetType[iPFJet] == 1:
                    quark_jets.append(pfjet)
                if evt.PFJetType[iPFJet] == 0:
                    gluon_jets.append(pfjet)
                    
    return quark_jets, gluon_jets



#loads input data
vbf_inputs = ROOT.TFile.Open('/afs/cern.ch/work/m/mmelodea/private/MonoHiggs/CMSSW_9_0_0/src/JetImageFiles/VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root')
ggh_inputs = ROOT.TFile.Open('/afs/cern.ch/work/m/mmelodea/private/MonoHiggs/CMSSW_9_0_0/src/JetImageFiles/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root')
inputs = [vbf_inputs,ggh_inputs]

#format tree inputs to adequate shape
quark_jets, gluon_jets = formatInputs(inputs)
print 'quark jets: %i' % len(quark_jets)
print 'gluon jets: %i' % len(gluon_jets)


#filter events based on DR(eta,phi)
maxDr = 0.4 #jet radius - ak4PFJetCHS
fquark_jets = []
fgluon_jets = []
max_entries = 1000

for ijet in quark_jets:
    if(ijet[0] < maxDr):
        fquark_jets.append(ijet)
	if(len(fquark_jets) >= max_entries):
	  break
        
for ijet in gluon_jets:
    if(ijet[0] < maxDr):
        fgluon_jets.append(ijet)
        if(len(fgluon_jets) >= max_entries):
          break
        
maxjets = min(len(fquark_jets),len(fgluon_jets))
fquark_jets = [fquark_jets[i] for i in range(maxjets)]
fgluon_jets = [fgluon_jets[i] for i in range(maxjets)]
print 'quark jets: %i (%.2f)' % (len(fquark_jets),len(fquark_jets)/float(len(quark_jets)))
print 'gluon jets: %i (%.2f)' % (len(fgluon_jets),len(fgluon_jets)/float(len(gluon_jets)))




#chose amount of pixels and pre-processing steps
nx = 51
xmin = -0.8
xmax = 0.8
ny = 51
ymin = -0.8
ymax = 0.8

#centralization, rotation, reflection, normalization
proc = [True,False,False,False]

#create the jet images
qjet_images, qlist_x, qlist_y, qlist_w, qxbins, qybins = prepareImages(fquark_jets, nx, xmin, xmax, ny, ymin, ymax, proc)
gjet_images, glist_x, glist_y, glist_w, gxbins, gybins = prepareImages(fgluon_jets, nx, xmin, xmax, ny, ymin, ymax, proc)




# Model
def build_conv_model():
    input_layer = Input(shape=(1, nx, ny))
    layer = Convolution2D(11, (20, 20), border_mode='same')(input_layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Convolution2D(11, (10, 10), border_mode='same')(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(3,3))(layer)
    layer = Convolution2D(11, (5, 5), border_mode='same')(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(3,3))(layer)
    layer = Flatten()(layer)
    layer = Dropout(0.20)(layer)
    layer = Dense(30)(layer)
    layer = Dropout(0.10)(layer)
    layer = Dense(15)(layer)
    output_layer = Dense(1, activation='sigmoid')(layer)
    model = Model(input=input_layer, output=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

conv_model = build_conv_model()
conv_model.summary()




#prepare inputs for model
X = np.concatenate([qjet_images, gjet_images])
qY = np.ones(qjet_images.shape[0])
gY = np.zeros(gjet_images.shape[0])
Y = np.concatenate([qY, gY])

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

kfold = StratifiedKFold(n_splits=2, shuffle=True,  random_state=seed)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)





# Train and plot ROC curves
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'black', 'green', 'brown'])
lw = 2

i = 0
histories = []
for (train, test), color in zip(kfold.split(X, encoded_Y), colors):
    conv_model = build_conv_model()
    history = conv_model.fit(X[train], encoded_Y[train], validation_data=(X[test], encoded_Y[test]), epochs=30, batch_size=128, verbose=1, callbacks=[early_stopping])
    Y_score = conv_model.predict(X[test])
    histories.append(history)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(encoded_Y[test], Y_score)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pyp.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1

mean_tpr /= kfold.get_n_splits(X, encoded_Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print 'ROC area: %.2f' % mean_auc

pyp.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
#pyp.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
#pyp.xlim([0, 1.0])
#pyp.ylim([0, 1.0])
#pyp.xlabel('False Positive Rate')
#pyp.ylabel('True Positive Rate')
#pyp.title('Receiver operating characteristic example')
#pyp.legend(loc="lower right")
#pyp.show()
