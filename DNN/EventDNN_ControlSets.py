#set adequate environment
import os
import sys
import theano
import keras
theano.config.gcc.cxxflags = '-march=corei7'
#load needed things
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.layers import Input, Activation, Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from scipy import interp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as pyp
import itertools
import math
import cPickle as pickle
from UserFunctions import *


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def CheckNetwork(infile,split_factor,use_vars,pre_process,wait_for,sbatch,lrate,layers,weights):
  #load events
  comparison = '>='
  njets = 2
  min4lmass = 118
  max4lmass = 130

  print "Loading dataset %s" % infile
  filein = open(infile,'r')
  events = pickle.load( filein )
  filein.close()

  print '----- Available MCs -----'
  print events.keys()
  print ''
  print '----- Available Variables -----'
  print events[events.keys()[0]].keys()


  #shows events and statistics
  print ''
  for ik in events:
    nevents = len(events[ik]['mc'])
    sumw = events[ik]['mc_sumweight'][0]
    print '%s events: %i, normalized: %.4f' % (ik,nevents,sumw)
    
  #only MC
  summ = 0
  for ik in events:
    if(ik != 'Data'):
        summ += len(events[ik]['mc'])
  print 'MC total events (absolute): %i' % summ

  print ''
  #organize events and splits mc into train, validation and test; data it's kept full
  #the split_factor sets train size then validation and test sets have the same size
  full_event_train, full_event_test, full_event_data = splitInputs(events, split_factor)

  #prepare train set
  X = {}
  Y = {}
  X['train'], Y['train'], Ydjet_train, Ymela_train, weights_train_set, scales_train_set = prepareSet(full_event_train, use_vars)

  #prepare test set
  X['test'], Y['test'], Ydjet_test, Ymela_test, weights_test_set, scales_test_set = prepareSet(full_event_test, use_vars)


  #before standardization #cross checking of inputs
  print "\nExample of inputs..."
  print X['train'][0]

  if(pre_process == 'norm'):
    #normalize inputs to [0, 1]
    X['train'] = preprocessing.normalize(X['train'])
    X['test'] = preprocessing.normalize(X['test'])
    print "After pre-processing..."
    print X['train'][0]
  
  if(pre_process == 'scale'):
    #standardize to have mean = 0 and sigma = 1
    X['train'] = preprocessing.scale(X['train'])
    X['test'] = preprocessing.scale(X['test'])
    print "After pre-processing..."
    print X['train'][0]


  # updatable plot
  class Tscheduler(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.logs = []
        self.trocs = []
        self.vrocs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        Y_score = dnn_model.predict(X['train'])
        fpr, tpr, thresholds = roc_curve(Y['train'], Y_score, sample_weight=weights_train_set)
        roc_auc = auc(fpr, tpr, reorder=True)
        self.trocs.append(roc_auc)
        Y_score = dnn_model.predict(X['test'])
        fpr, tpr, thresholds = roc_curve(Y['test'], Y_score, sample_weight=weights_test_set)
        roc_auc = auc(fpr, tpr, reorder=True)
        self.vrocs.append(roc_auc)
        self.i += 1
        print "epoch: %i, tloss: %.2f, vloss: %.2f, troc: %.3f, vroc: %.3f" % (self.x,self.losses[len(self.losses)-1],self.val_losses[len(self.val_losses)-1],self.trocs[len(self.trocs)-1],self.vrocs[len(self.vrocs)-1])
            
  Tschedule = Tscheduler()


  #parameters for training
  nepochs = 10000
  opt = Adam()
  if(lrate != -1):
    opt = Adam(lr=lrate)
  early_stopping = EarlyStopping(monitor='val_loss', patience=wait_for)


  #DNN network
  #from keras.layers import Dropout
  ninputs = len(X['train'][0])
  print "\n---------- DNN Topology --------- "
  print "Inputs: %i" % ninputs

  dnn_model = Sequential()
  dnn_model.add(Dense(layers[0], input_shape=(ninputs,), activation='relu', kernel_initializer='random_uniform'))
  for ilayer in range(1,len(layers)):
    dnn_model.add(Dense(layers[ilayer], activation='relu', kernel_initializer='random_uniform'))    
  dnn_model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))    
  dnn_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  print dnn_model.summary()

  filepath="best_weights.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

  #train the DNN
  history = []
  if(weights == 'sum'):
    history = dnn_model.fit(X['train'],Y['train'],sample_weight=scales_train_set,
		  validation_data=(X['test'],Y['test'],scales_test_set), 
		  epochs=nepochs,batch_size=sbatch,verbose=0,
		  callbacks=[checkpoint, early_stopping]#,Tschedule]
		 )
  if(weights == 'individual'):
    history = dnn_model.fit(X['train'],Y['train'],sample_weight=weights_train_set,
		  validation_data=(X['test'],Y['test'],weights_test_set), 
		  epochs=nepochs,batch_size=sbatch,verbose=0,
		  callbacks=[checkpoint, early_stopping]#,Tschedule]
		 )


  pyp.rc("font", size=18)
  fig = pyp.figure(figsize=(10,10))
  #plot training history
  val_loss = np.asarray(history.history['val_loss'])
  loss = np.asarray(history.history['loss'])
  pyp.plot(loss, label='loss train')
  pyp.plot(val_loss, label='loss validation')
  pyp.legend()
  pyp.xlabel('epoch')
  pyp.grid(True)
  pyp.savefig("TrainingHistory.png")
  

  # load final best weights
  dnn_model.load_weights(filepath)

  #train set normalized roc
  Y_score = dnn_model.predict(X['train'])
  fpr, tpr, thresholds = roc_curve(Y['train'], Y_score, sample_weight=weights_train_set)
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', color='blue', label='DNN train AUC = %0.3f' % roc_auc)
  print '\nDNN train AUC = %0.3f' % roc_auc

  #test set unormalized roc
  Y_score = dnn_model.predict(X['test'])
  fpr, tpr, thresholds = roc_curve(Y['test'], Y_score, sample_weight=weights_test_set)
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, color='red', label='DNN test AUC = %0.3f' % roc_auc)
  print 'DNN test AUC = %0.3f' % roc_auc

  #random
  pyp.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck')
  
  pyp.xlim([0, 1.0])
  pyp.ylim([0, 1.0])
  pyp.xlabel('False Positive Rate')
  pyp.ylabel('True Positive Rate')
  pyp.title('Receiver Operating Characteristic')
  pyp.legend(loc="lower right")
  pyp.grid(True)
  pyp.savefig("ROCsComparisonTrainTest.png")



  fig = pyp.figure(figsize=(10,10))
  
  #normalized rocs
  fpr, tpr, thresholds = roc_curve(Y['test'], Ydjet_test, sample_weight=weights_test_set)
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', lw=2, color='blue', label=('Djet test AUC = %0.3f' % roc_auc))
  print '\nDjet test AUC = %0.3f' % roc_auc

  fpr, tpr, thresholds = roc_curve(Y['test'], Ymela_test, sample_weight=weights_test_set)
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', lw=2, color='red', label=('MELA test AUC = %0.3f' % roc_auc))
  print 'MELA test AUC = %0.3f' % roc_auc

  Y_score = dnn_model.predict(X['test'])
  fpr, tpr, thresholds = roc_curve(Y['test'], Y_score, sample_weight=weights_test_set)
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', lw=2, color='green', label=('DNN test AUC = %0.3f' % roc_auc))
  print 'DNN test AUC = %0.3f' % roc_auc

  pyp.xlim([0, 1.0])
  pyp.ylim([0, 1.0])
  pyp.xlabel('False Positive Rate')
  pyp.ylabel('True Positive Rate')
  pyp.title('Receiver Operating Characteristic')
  pyp.legend(loc="lower right")
  pyp.grid(True)
  pyp.savefig("ROCsComparisonDiscriminants.png")


  #Make some plots to compare the discriminants
  signal_sumw = 0
  for iev in range(len(full_event_test['VBF']['mc'])):
    signal_sumw += full_event_test['VBF']['event_weight'][iev]

  djet_sb = []
  djet_seff = []
  djet_ep = []
  mela_sb = []
  mela_seff = []
  mela_ep = []
  dnn_sb = []
  dnn_seff = []
  dnn_ep = []
  djet_cuts = []
  mela_cuts = []
  dnn_cuts = []

  efull_sdjet = []
  efull_bdjet = []
  efull_smela = []
  efull_bmela = []
  efull_sdnn = []
  efull_bdnn = []
  esweights = []
  ebweights = []
  
  for cut in range(300):
    icut = cut/100.
    
    s_djet = 0
    s_mela = 0
    s_dnn = 0
    b_djet = 0
    b_mela = 0
    b_dnn = 0
    for i in range(len(Y['test'])):
        djet = Ydjet_test[i]
        mela = Ymela_test[i]
        dnn = Y_score[i][0]
        weight = weights_test_set[i]
        
        #VBF (signal)
        if(Y['test'][i] == 1):
            if(cut == 0):
                efull_sdjet.append( djet )
                efull_smela.append( mela )
                efull_sdnn.append( dnn )
                esweights.append( weight )
            
            if(djet > icut):
                s_djet += weight
            if(mela > icut):
                s_mela += weight
            if(dnn > icut):
                s_dnn += weight
        
        #Backgrounds
        else:
            if(cut == 0):
                efull_bdjet.append( djet )
                efull_bmela.append( mela )
                efull_bdnn.append( dnn )
                ebweights.append( weight )
            
            if(djet > icut):
                b_djet += weight
            if(mela > icut):
                b_mela += weight
            if(dnn > icut):
                b_dnn += weight
    
    if(s_djet > 0 or b_djet > 0):
        djet_ep.append( (s_djet/signal_sumw)*(s_djet/(s_djet+b_djet)) )
        djet_sb.append(s_djet/math.sqrt(b_djet+s_djet))
        djet_seff.append(s_djet/signal_sumw)
        djet_cuts.append(icut)

    if(icut <= 1 and (s_mela > 0 or b_mela > 0)):
        mela_ep.append( (s_mela/signal_sumw)*(s_mela/(s_mela+b_mela)) )
        mela_sb.append(s_mela/math.sqrt(b_mela+s_mela))
        mela_seff.append(s_mela/signal_sumw)
        mela_cuts.append(icut)
        
    if(icut <= 1 and (s_dnn > 0 or b_dnn > 0)):
        dnn_ep.append( (s_dnn/signal_sumw)*(s_dnn/(s_dnn+b_dnn)) )
        dnn_sb.append(s_dnn/math.sqrt(b_dnn+s_dnn))
        dnn_seff.append(s_dnn/signal_sumw)
        dnn_cuts.append(icut)

  index = np.argmax(mela_ep)
  print 'Max MELA Seff_purity = %.3f, Seff = %.3f, S/sqrt(S+B) = %.3f at cut = %.3f' % (mela_ep[index], mela_seff[index], mela_sb[index], mela_cuts[index])
  index = np.argmax(dnn_ep)
  print 'Max DNN Seff_purity = %.3f, Seff = %.3f, S/sqrt(S+B) = %.3f at cut = %.3f' % (dnn_ep[index], dnn_seff[index], dnn_sb[index], dnn_cuts[index])
  
  #compute the same quantities above for training set
  tfull_sdjet = []
  tfull_bdjet = []
  tfull_smela = []
  tfull_bmela = []
  tfull_sdnn = []
  tfull_bdnn = []
  tsweights = []
  tbweights = []

  TrainScore = dnn_model.predict(X['train'])
  for iev in range(len(Y['train'])):
    djet = Ydjet_train[iev]
    mela = Ymela_train[iev]
    dnn = TrainScore[iev][0]
    weight = weights_train_set[iev]

    if(Y['train'][iev] == 1):
        tfull_sdjet.append( djet )
        tfull_smela.append( mela )
        tfull_sdnn.append( dnn )
        tsweights.append( weight )
    else:
        tfull_bdjet.append( djet )
        tfull_bmela.append( mela )
        tfull_bdnn.append( dnn )
        tbweights.append( weight )



  #current metric we adopted
  fig = pyp.figure()
  fig.set_figheight(7)
  fig.set_figwidth(14)

  fig1 = fig.add_subplot(121)
  pyp.plot(djet_seff, djet_ep, lw=1, color='blue', label='Djet: %.3f'% auc(djet_seff, djet_ep,reorder=True))
  pyp.plot(mela_seff, mela_ep, lw=1, color='red', label='MELA: %.3f'% auc(mela_seff, mela_ep,reorder=True))
  pyp.plot(dnn_seff, dnn_ep, lw=1, color='green', label='DNN: %.3f'% auc(dnn_seff, dnn_ep,reorder=True))
  pyp.xlim([0, 1])
  pyp.xlabel('Signal eff')
  pyp.ylim([0, 1])
  pyp.ylabel('eff*purity')
  pyp.legend()
  pyp.grid(True)

  fig2 = fig.add_subplot(122)
  pyp.plot(djet_seff, djet_sb, lw=1, color='blue', label='Djet: %.3f'% auc(djet_seff, djet_sb,reorder=True))
  pyp.plot(mela_seff, mela_sb, lw=1, color='red', label='MELA: %.3f'% auc(mela_seff, mela_sb,reorder=True))
  pyp.plot(dnn_seff, dnn_sb, lw=1, color='green', label='DNN: %.3f'% auc(dnn_seff, dnn_sb,reorder=True))
  pyp.xlim([0, 1])
  pyp.xlabel('Signal eff')
  #pyp.ylim([0, 1])
  pyp.ylabel('S/sqrt(S+B)')
  pyp.legend()
  pyp.grid(True)

  pyp.tight_layout()
  fig.savefig("MetricsComparisonDiscriminants.png")


  ##### ---------------- Cross checking for overfitting ------------------ ####
  fig = pyp.figure()
  fig.set_figheight(5)
  fig.set_figwidth(14)
  import ROOT

  ###### Djet  
  fig1 = fig.add_subplot(131)
  th_tsdjet = ROOT.TH1D('1','',60,0,3)
  th_tsdjet.Sumw2()
  centers1 = []
  counts1 = []
  err1 = []
  for i in range(len(tfull_sdjet)):
    th_tsdjet.Fill(tfull_sdjet[i], tsweights[i])
  th_tsdjet.Scale(1./th_tsdjet.Integral(), 'width')
  for ib in range(th_tsdjet.GetNbinsX()):
    centers1.append( th_tsdjet.GetBinCenter(ib+1) )
    counts1.append( th_tsdjet.GetBinContent(ib+1) )
    err1.append( th_tsdjet.GetBinError(ib+1) )
  
  th_tbdjet = ROOT.TH1D('2','',60,0,3)
  th_tbdjet.Sumw2()
  centers2 = []
  counts2 = []
  err2 = []
  for i in range(len(tfull_bdjet)):
    th_tbdjet.Fill(tfull_bdjet[i], tbweights[i])
  th_tbdjet.Scale(1./th_tbdjet.Integral(), 'width')    
  for ib in range(th_tbdjet.GetNbinsX()):
    centers2.append( th_tbdjet.GetBinCenter(ib+1) )
    counts2.append( th_tbdjet.GetBinContent(ib+1) )
    err2.append( th_tbdjet.GetBinError(ib+1) )
  
  th_esdjet = ROOT.TH1D('3','',60,0,3)
  th_esdjet.Sumw2()
  centers3 = []
  counts3 = []
  err3 = []
  for i in range(len(efull_sdjet)):
    th_esdjet.Fill(efull_sdjet[i], esweights[i])  
  th_esdjet.Scale(1./th_esdjet.Integral(), 'width')
  for ib in range(th_esdjet.GetNbinsX()):
    centers3.append( th_esdjet.GetBinCenter(ib+1) )
    counts3.append( th_esdjet.GetBinContent(ib+1) )
    err3.append( th_esdjet.GetBinError(ib+1) )
  
  th_ebdjet = ROOT.TH1D('4','',60,0,3)
  th_ebdjet.Sumw2()
  centers4 = []
  counts4 = []
  err4 = []
  for i in range(len(efull_bdjet)):
    th_ebdjet.Fill(efull_bdjet[i], ebweights[i])  
  th_ebdjet.Scale(1./th_ebdjet.Integral(), 'width')
  for ib in range(th_ebdjet.GetNbinsX()):
    centers4.append( th_ebdjet.GetBinCenter(ib+1) )
    counts4.append( th_ebdjet.GetBinContent(ib+1) )
    err4.append( th_ebdjet.GetBinError(ib+1) )
  
  pyp.errorbar(centers1, counts1, yerr=err1, fmt='bo', markersize=1, elinewidth=1)
  pyp.hist(centers1, np.linspace(0, 3, 61), weights=counts1, histtype='stepfilled', alpha=0.3, color='b', label = 's-train')
  pyp.errorbar(centers2, counts2, yerr=err2, fmt='ro', markersize=1, elinewidth=1)
  pyp.hist(centers2, np.linspace(0, 3, 61), weights=counts2, histtype='stepfilled', alpha=0.3, color='r', label = 'b-train')
  pyp.errorbar(centers3, counts3, yerr=err3, fmt='bo', markersize=4, elinewidth=1, label = 's-test')
  pyp.hist(centers3, np.linspace(0, 3, 61), weights=counts3, histtype='step', color='b', lw=1)
  pyp.errorbar(centers4, counts4, yerr=err4, fmt='ro', markersize=4, elinewidth=1, label = 'b-test')
  pyp.hist(centers4, np.linspace(0, 3, 61), weights=counts4, histtype='step', color='r', lw=1)
  pyp.xlabel('Djet')
  pyp.ylabel('Normalized')
  #pyp.ylim([0.002,100])
  pyp.legend()
  #pyp.gca().set_yscale("log")

  ######## MELA
  fig2 = fig.add_subplot(132)
  th_tsmela = ROOT.TH1D('5','',40,0,1)
  th_tsmela.Sumw2()
  centers1 = []
  counts1 = []
  err1 = []
  for i in range(len(tfull_smela)):
    th_tsmela.Fill(tfull_smela[i], tsweights[i])
  th_tsmela.Scale(1./th_tsmela.Integral(), 'width')
  for ib in range(th_tsmela.GetNbinsX()):
    centers1.append( th_tsmela.GetBinCenter(ib+1) )
    counts1.append( th_tsmela.GetBinContent(ib+1) )
    err1.append( th_tsmela.GetBinError(ib+1) )
  
  th_tbmela = ROOT.TH1D('6','',40,0,1)
  th_tbmela.Sumw2()
  centers2 = []
  counts2 = []
  err2 = []
  for i in range(len(tfull_bmela)):
    th_tbmela.Fill(tfull_bmela[i], tbweights[i])
  th_tbmela.Scale(1./th_tbmela.Integral(), 'width')    
  for ib in range(th_tbmela.GetNbinsX()):
    centers2.append( th_tbmela.GetBinCenter(ib+1) )
    counts2.append( th_tbmela.GetBinContent(ib+1) )
    err2.append( th_tbmela.GetBinError(ib+1) )
  
  th_esmela = ROOT.TH1D('7','',40,0,1)
  th_esmela.Sumw2()
  centers3 = []
  counts3 = []
  err3 = []
  for i in range(len(efull_smela)):
    th_esmela.Fill(efull_smela[i], esweights[i])  
  th_esmela.Scale(1./th_esmela.Integral(), 'width')
  for ib in range(th_esmela.GetNbinsX()):
    centers3.append( th_esmela.GetBinCenter(ib+1) )
    counts3.append( th_esmela.GetBinContent(ib+1) )
    err3.append( th_esmela.GetBinError(ib+1) )
  
  th_ebmela = ROOT.TH1D('8','',40,0,1)
  th_ebmela.Sumw2()
  centers4 = []
  counts4 = []
  err4 = []
  for i in range(len(efull_bmela)):
    th_ebmela.Fill(efull_bmela[i], ebweights[i])  
  th_ebmela.Scale(1./th_ebmela.Integral(), 'width')
  for ib in range(th_ebmela.GetNbinsX()):
    centers4.append( th_ebmela.GetBinCenter(ib+1) )
    counts4.append( th_ebmela.GetBinContent(ib+1) )
    err4.append( th_ebmela.GetBinError(ib+1) )
  
  pyp.errorbar(centers1, counts1, yerr=err1, fmt='bo', markersize=1, elinewidth=1)
  pyp.hist(centers1, np.linspace(0, 1, 41), weights=counts1, histtype='stepfilled', alpha=0.3, color='b', label = 's-train')
  pyp.errorbar(centers2, counts2, yerr=err2, fmt='ro', markersize=1, elinewidth=1)
  pyp.hist(centers2, np.linspace(0, 1, 41), weights=counts2, histtype='stepfilled', alpha=0.3, color='r', label = 'b-train')
  pyp.errorbar(centers3, counts3, yerr=err3, fmt='bo', markersize=4, elinewidth=1, label = 's-test')
  pyp.hist(centers3, np.linspace(0, 1, 41), weights=counts3, histtype='step', color='b', lw=1)
  pyp.errorbar(centers4, counts4, yerr=err4, fmt='ro', markersize=4, elinewidth=1, label = 'b-test')
  pyp.hist(centers4, np.linspace(0, 1, 41), weights=counts4, histtype='step', color='r', lw=1)
  pyp.xlabel('MELA')
  #pyp.ylim([0.002,100])
  pyp.legend()
  #pyp.gca().set_yscale("log")

  fig3 = fig.add_subplot(133)
  th_tsdnn = ROOT.TH1D('9','',40,0,1)
  th_tsdnn.Sumw2()
  centers1 = []
  counts1 = []
  err1 = []
  for i in range(len(tfull_sdnn)):
    th_tsdnn.Fill(tfull_sdnn[i], tsweights[i])
  th_tsdnn.Scale(1./th_tsdnn.Integral(), 'width')
  for ib in range(th_tsdnn.GetNbinsX()):
    centers1.append( th_tsdnn.GetBinCenter(ib+1) )
    counts1.append( th_tsdnn.GetBinContent(ib+1) )
    err1.append( th_tsdnn.GetBinError(ib+1) )
  
  th_tbdnn = ROOT.TH1D('10','',40,0,1)
  th_tbdnn.Sumw2()
  centers2 = []
  counts2 = []
  err2 = []
  for i in range(len(tfull_bdnn)):
    th_tbdnn.Fill(tfull_bdnn[i], tbweights[i])
  th_tbdnn.Scale(1./th_tbdnn.Integral(), 'width')    
  for ib in range(th_tbdnn.GetNbinsX()):
    centers2.append( th_tbdnn.GetBinCenter(ib+1) )
    counts2.append( th_tbdnn.GetBinContent(ib+1) )
    err2.append( th_tbdnn.GetBinError(ib+1) )
  
  th_esdnn = ROOT.TH1D('11','',40,0,1)
  th_esdnn.Sumw2()
  centers3 = []
  counts3 = []
  err3 = []
  for i in range(len(efull_sdnn)):
    th_esdnn.Fill(efull_sdnn[i], esweights[i])  
  th_esdnn.Scale(1./th_esdnn.Integral(), 'width')
  for ib in range(th_esdnn.GetNbinsX()):
    centers3.append( th_esdnn.GetBinCenter(ib+1) )
    counts3.append( th_esdnn.GetBinContent(ib+1) )
    err3.append( th_esdnn.GetBinError(ib+1) )
  
  th_ebdnn = ROOT.TH1D('12','',40,0,1)
  th_ebdnn.Sumw2()
  centers4 = []
  counts4 = []
  err4 = []
  for i in range(len(efull_bdnn)):
    th_ebdnn.Fill(efull_bdnn[i], ebweights[i])  
  th_ebdnn.Scale(1./th_ebdnn.Integral(), 'width')
  for ib in range(th_ebdnn.GetNbinsX()):
    centers4.append( th_ebdnn.GetBinCenter(ib+1) )
    counts4.append( th_ebdnn.GetBinContent(ib+1) )
    err4.append( th_ebdnn.GetBinError(ib+1) )
  
  pyp.errorbar(centers1, counts1, yerr=err1, fmt='bo', markersize=1, elinewidth=1)
  pyp.hist(centers1, np.linspace(0, 1, 41), weights=counts1, histtype='stepfilled', alpha=0.3, color='b', label = 's-train')
  pyp.errorbar(centers2, counts2, yerr=err2, fmt='ro', markersize=1, elinewidth=1)
  pyp.hist(centers2, np.linspace(0, 1, 41), weights=counts2, histtype='stepfilled', alpha=0.3, color='r', label = 'b-train')
  pyp.errorbar(centers3, counts3, yerr=err3, fmt='bo', markersize=4, elinewidth=1, label = 's-test')
  pyp.hist(centers3, np.linspace(0, 1, 41), weights=counts3, histtype='step', color='b', lw=1)
  pyp.errorbar(centers4, counts4, yerr=err4, fmt='ro', markersize=4, elinewidth=1, label = 'b-test')
  pyp.hist(centers4, np.linspace(0, 1, 41), weights=counts4, histtype='step', color='r', lw=1)
  pyp.xlabel('DNN')
  #pyp.ylim([0.002,100])
  pyp.legend()
  #pyp.gca().set_yscale("log")

  pyp.tight_layout()
  fig.savefig("DiscriminantsDistributions.png")



  #creates normalized ROCs separated by MC
  print '\n******** ROCs from DNN *********'
  fig = pyp.figure(figsize=(10,10))
  colors = ['red','blue','violet','green','orange','yellow','cyan']

  Y_truth = {}
  Y_mela = {}
  X = {}
  Weights = {}
  for ik in full_event_test:
    Y_truth[ik] = []
    X[ik] = []
    Weights[ik] = []
    Y_mela[ik] = []

    for iev in range(len(full_event_test[ik]['mc'])):
        variables = []
        for ivar in use_vars:
            variables.append(full_event_test[ik][ivar][iev])
        X[ik].append(variables)
        if(pre_process == 'norm'):
          #normalize inputs to [0, 1]
          X[ik] = preprocessing.normalize(X[ik])
        if(pre_process == 'scale'):
          #standardize to have mean = 0 and sigma = 1
          X[ik] = preprocessing.scale(X[ik])

        if(ik == 'VBF'):
            Y_truth[ik].append(1)
        else:
            Y_truth[ik].append(0)
        Weights[ik].append(full_event_test[ik]['event_weight'][iev])
        
  if(pre_process == 'norm' or pre_process == 'scale'):
      inputs = []
      weights = []
      for ik in X:
        for i in range(len(X[ik])):
          inputs.append( X[ik][i] )
          weights.append( Weights[ik][i] )
      if(pre_process == 'norm'):
        inputs = preprocessing.normalize(inputs)
      if(pre_process == 'scale'):
        inputs = preprocessing.scale(inputs)
      index = 0
      for ik in X:
        for i in range(len(X[ik])):
	  X[ik][i] = inputs[index]
	  Weights[ik][i] = weights[index]
	  index += 1
        
                
  ic = 0
  for ik in Y_truth:
    if(ik == 'VBF'):
        continue
    cX = np.concatenate([X['VBF'],X[ik]])
    cY = np.concatenate([Y_truth['VBF'],Y_truth[ik]])
    cW = np.concatenate([Weights['VBF'],Weights[ik]])
    Y_score = dnn_model.predict(cX)
    fpr, tpr, thresholds = roc_curve(cY, Y_score, sample_weight=cW)
    roc_auc = auc(fpr, tpr,reorder=True)
    print 'VBF vs ',ik,' -- ROC AUC: %.2f' % roc_auc
    pyp.plot(fpr, tpr, lw=1, color=colors[ic], label=('VBF vs %s (%0.2f)' % (ik,roc_auc)))
    ic += 1
    
  #------------------------------------------------------------------------------------------------------------
  print ''
  print '******** ROCs from MELA *********'
  for ik in full_event_test:
    for iev in range(len(full_event_test[ik]['mc'])):    
        Y_mela[ik].append(full_event_test[ik]['DjetVAJHU'][iev])

  ic = 0
  for ik in Y_truth:
    if(ik == 'VBF'):
        continue
    sY = np.concatenate([Y_mela['VBF'],Y_mela[ik]])
    cY = np.concatenate([Y_truth['VBF'],Y_truth[ik]])
    cW = np.concatenate([Weights['VBF'],Weights[ik]])
    fpr, tpr, thresholds = roc_curve(cY, sY, sample_weight=cW)
    roc_auc = auc(fpr, tpr,reorder=True)
    print 'VBF vs ',ik,' -- ROC AUC: %.2f' % roc_auc
    pyp.plot(fpr, tpr, linestyle='--', lw=1, color=colors[ic], label=('VBF vs %s (%0.2f)' % (ik,roc_auc)))
    ic += 1
    
  pyp.xlim([0.001, 1.0])
  pyp.ylim([0.001, 1.0])
  pyp.xlabel('False Positive Rate')
  pyp.ylabel('True Positive Rate')
  #pyp.xscale('log')
  #pyp.yscale('log')
  pyp.title('Receiver Operating Characteristic')
  pyp.legend(loc="lower right")
  pyp.grid(True)
  pyp.savefig("SeparatedROCsComparisonMELAtoNetwork.png")
  
#----------- End of Train/Test function




#main
infile = 'hzz4l_vbf_selection_noDjet_m4l118-130GeV_shuffledFS.pkl'
split_factor = _split_factor
use_vars = [_use_vars]
pre_process = _pre_process
wait_for = _wait_for
sbatch = _sbatch
lrate = _lrate
layers = [_layers]
weights = _use_weights

print "\n---------------------- STARTING -----------------------"
print ">>>>>>>>>>>>>>>>>>>>>> Configuration <<<<<<<<<<<<<<<<<<<<"
print "split_factor: ", split_factor
print "use_vars: ", use_vars
print "pre_process: ", pre_process
print "wait_for: ", wait_for
print "sbatch: ", sbatch
print "lrate: ", lrate
print "layers: ", layers
print "weights: ", weights
print "----------------------------------------------------------"
print "----------------------- INITIATING -----------------------"
CheckNetwork(infile,split_factor,use_vars,pre_process,wait_for,sbatch,lrate,layers,weights)
print "\n---------------------- FINALIZED -----------------------"
