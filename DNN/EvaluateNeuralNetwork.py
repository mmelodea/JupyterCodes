###------------------------------------------------------------------------------------###
###  This code runs over FormatROOTs.py dictionary                                     ###
###  It allows a fast Keras Dense NN building, training and testing                    ###
###  Author: Miqueias Melo de Almeida                                                  ###
###  Data: 05/06/2018                                                                  ###
###------------------------------------------------------------------------------------###

#set adequate environment
import os
import sys
import argparse
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
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from scipy import interp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as pyp
from matplotlib import gridspec
import itertools
import math
import ROOT
import cPickle as pickle


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#### function to organize events and splits data in train, test and real data sets ###
def prepareSets(events, split_factor, use_vars, use_mcs, signal, nooutliers):
  import numpy as np
  
  #### filter out outliers if requested
  if nooutliers:
    print 'Cleaning outliers...'
    cleaned_events = {}
    for ik in use_mcs:
      cleaned_events[ik] = {}
      nevents = len(events[ik]['mc'])
      for ivar in events[ik]:
	cleaned_events[ik][ivar] = []
	for iev in range(nevents):
	  if not events[ik]['f_outlier'][iev]:
	    cleaned_events[ik][ivar].append(events[ik][ivar][iev])
    #re-assign events content
    events = cleaned_events

  ntrain = 0
  ntest = 0
  default_value = 0
  #dictionary to contains train and test set
  inputs = {}
  inputs['train'] = {}
  inputs['test'] = {}
        
  for ik in use_mcs:       
    inputs['train'][ik] = {}
    inputs['test'][ik] = {}       
    #print ik,' = ',nevents,' events'
    for ivar in events[ik]:
      #prepare things for training set
      inputs['train'][ik][ivar] = []
      #prepare things for testing set
      inputs['test'][ik][ivar] = []
           
      nevents = len(events[ik]['mc'])
      for iev in range(nevents):
        #---- fills the training set ----
        if(iev < int(nevents*split_factor)):
          inputs['train'][ik][ivar].append( events[ik][ivar][iev] )
          #to count just once - gets the number of events got
          if(ivar == 'mc'):
            ntrain += 1
        #---- fills the testing set -----
        else:
          inputs['test'][ik][ivar].append( events[ik][ivar][iev] )
          #to count just once - gets the number of events got
          if(ivar == 'mc'):
            ntest += 1
    
  #prepare the final vectors
  finputs = {}
  flabels = {}
  fweights = {}
  fscales = {}
  fmela = {}
  for iset in inputs:
    print '------ Set = ',iset,' ------'
    finputs[iset] = []
    flabels[iset] = []
    fweights[iset] = []
    fscales[iset] = []
    fmela[iset] = []
    for ik in inputs[iset]:
      nevents = len(inputs[iset][ik]['mc'])
      print ik,' = ',nevents
      for iev in range(nevents):
        if(ik in signal):
          flabels[iset].append( 1 )
        else:
          flabels[iset].append( 0 )
                    
        fweights[iset].append( inputs[iset][ik]['f_weight'][iev] )
        fscales[iset].append( inputs[iset][ik]['mc_sumweight'][iev] )
        fmela[iset].append( inputs[iset][ik]['f_Djet_VAJHU'][iev] )
        variables = []
        for ivar in use_vars:
          variables.append( inputs[iset][ik][ivar][iev] )
        finputs[iset].append( variables )
    print ">>> Size of",iset," set  = ",len(finputs[iset])

    
  #converts to numpy array format (needed for Keras)
  tinputs = np.asarray(finputs['train'])
  tlabels = np.asarray(flabels['train'])
  tweights = np.asarray(fweights['train'])
  tscales = np.asarray(fscales['train'])
  vinputs = np.asarray(finputs['test'])
  vlabels = np.asarray(flabels['test'])
  vweights = np.asarray(fweights['test'])
  vscales = np.asarray(fscales['test'])
    
  #create final dictionaries
  d_inputs = {'train':tinputs, 'test':vinputs}
  d_labels = {'train':tlabels, 'test':vlabels}
  d_mela = {'train':fmela['train'], 'test':fmela['test']}
  d_weights = {'train':tweights, 'test':vweights}
  d_scales = {'train':tscales, 'test':vscales}
    
  return d_inputs, d_labels, d_mela, d_weights, d_scales


###------------------------------------------- Build, Train and Test NN ------------------------------------------------###
def TrainNeuralNetwork(filein_name, results_folder, use_mcs, signal, use_vars, split_factor, pre_proc, layers, neuron, nepochs, wait_for, sbatch, opt, scale_train, nooutliers):
  #### creates a dictionary to hold informations
  outdict = {}
  outdict['infile'] = filein_name
  outdict['signal'] = signal
  outdict['nninputs'] = use_vars
  outdict['split'] = split_factor
  outdict['topology'] = layers
  outdict['neuron'] = neuron
  outdict['epochs'] = nepochs
  outdict['patience'] = wait_for
  outdict['batchsize'] = sbatch
  outdict['minimizer'] = opt
  outdict['scaletrain'] = scale_train
  outdict['nooutliers'] = nooutliers

  print '>>>>> Results will be saved there: ',results_folder
  if not os.path.isdir(results_folder):
    os.mkdir(results_folder)
  
  print "Loading file: ",filein_name
  filein = open(filein_name,'r')
  events = pickle.load( filein )
  filein.close()

  #print ''
  print '----- Available MCs -----'
  print events.keys()
  print ''
  if(use_mcs == None):
    use_mcs = events.keys()

  #shows events and statistics
  outdict['mcs'] = use_mcs
  for ik in use_mcs:
    nevents = len(events[ik]['mc'])
    #print 'nevents = ',nevents,' :: ',events[ik]['mc_sumweight']
    sumw = events[ik]['mc_sumweight'][0]
    print '%s events: %i, normalized: %.4f' % (ik,nevents,sumw)
    outdict['mcinfo'] = {ik:[nevents,sumw]}
    

  #prepare train set
  X = {}
  Y = {}
  X, Y, Ymela, weights, scales = prepareSets(events, split_factor, use_vars, use_mcs, signal, nooutliers)
  outdict['trainsize'] = len(Y['train'])
  outdict['testsize'] = len(Y['test'])

  #before standardization #cross checking of inputs
  print ''
  print "-------- Overview of Inputs ---------"
  print X['train']

  if(pre_proc == 'normalize'):
    #normalize inputs to [0, 1]
    X['train'] = preprocessing.normalize(X['train'])
    X['test'] = preprocessing.normalize(X['test'])
    print "----------- After Normalizing ------------"
    print X['train'][0]

  elif(pre_proc == 'scale'):
    #standardize to have mean = 0 and sigma = 1
    X['train'] = preprocessing.scale(X['train'])
    X['test'] = preprocessing.scale(X['test'])
    print "---------- After Scaling -----------------"
    print X['train'][0]

  else:
    print ">>> No pre-processing applied to inputs!"


  #Building Network
  ninputs = len(X['train'][0])
  print "---------- DNN Topology --------- "
  print "Inputs: %i" % ninputs

  model = Sequential()
  model.add(Dense(layers[0], input_shape=(ninputs,), activation=neuron[0], kernel_initializer='random_uniform'))
  for ilayer in range(1,len(layers)):
    model.add(Dense(layers[ilayer], activation=neuron[0], kernel_initializer='random_uniform'))
  model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))    
  
  if(opt[0] == 'sgd'):
    model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
  if(opt[0] == 'adam'):
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
  if(opt[0] == 'adagrad'):
    model.compile(loss='binary_crossentropy', optimizer=Adagrad(), metrics=['accuracy'])
  if(opt[0] == 'adadelta'):
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
  if(opt[0] == 'rmsprop'):
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    
  print model.summary()

  filepath=results_folder+"/best_model.h5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
  early_stopping = EarlyStopping(monitor='val_loss', patience=wait_for)

  #train the DNN
  history = []
  if(scale_train[0] == 'none'):
    history = model.fit(X['train'],Y['train'],validation_data=(X['test'],Y['test']),
			epochs=nepochs,batch_size=sbatch,verbose=2,callbacks=[checkpoint, early_stopping])    
  if(scale_train[0] == 'mc_weight'):
    history = model.fit(X['train'],Y['train'],sample_weight=scales['train'],validation_data=(X['test'],Y['test'],scales['test']),
			epochs=nepochs,batch_size=sbatch,verbose=2,callbacks=[checkpoint, early_stopping])
  if(scale_train[0] == 'event_weight'):
    history = model.fit(X['train'],Y['train'],sample_weight=weights['train'],validation_data=(X['test'],Y['test'],weights['test']), 
                         epochs=nepochs,batch_size=sbatch,verbose=2,callbacks=[checkpoint, early_stopping])
			

  print ''
  print '----------- Training Finished --------------------'
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
  pyp.savefig(results_folder+"/TrainingHistoryLoss.png")

  # load final best weights
  model.load_weights(filepath)
  

  #compute accuracy
  print ''
  print '-------------- Accuracy ----------------------'
  tscores = model.evaluate(X['train'], Y['train'])
  print("(On TRAIN set)%s: %.2f%%" % (model.metrics_names[1], tscores[1]*100))
  vscores = model.evaluate(X['test'], Y['test'])
  print("(On TEST set)%s: %.2f%%" % (model.metrics_names[1], vscores[1]*100))
  outdict['accuracy'] = {'train':tscores[1]*100,'test':vscores[1]*100}

  pyp.rc("font", size=18)
  fig = pyp.figure(figsize=(10,10))

  #normalized rocs
  outdict['mela'] = {}
  fpr, tpr, thresholds = roc_curve(Y['train'], Ymela['train'], sample_weight=weights['train'])
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', lw=1, color='red', label=('MELA train AUC = %0.3f' % roc_auc))
  outdict['mela']['roctrain'] = roc_auc

  fpr, tpr, thresholds = roc_curve(Y['test'], Ymela['test'], sample_weight=weights['test'])
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='-', lw=2, color='red', label=('MELA test AUC = %0.3f' % roc_auc))
  outdict['mela']['roctest'] = roc_auc
  
  outdict['nn'] = {}
  Predictions = {}
  Predictions['train'] = model.predict(X['train'])
  fpr, tpr, thresholds = roc_curve(Y['train'], Predictions['train'], sample_weight=weights['train'])
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='--', lw=1, color='blue', label='DNN train AUC = %0.3f' % roc_auc)
  outdict['nn']['roctrain'] = roc_auc
  
  Predictions['test'] = model.predict(X['test'])
  fpr, tpr, thresholds = roc_curve(Y['test'], Predictions['test'], sample_weight=weights['test'])
  roc_auc = auc(fpr, tpr,reorder=True)
  pyp.plot(fpr, tpr, linestyle='-', lw=2, color='blue', label=('DNN test AUC = %0.3f' % roc_auc))
  outdict['nn']['roctest'] = roc_auc
  
  pyp.xlim([0, 1.0])
  pyp.ylim([0, 1.0])
  pyp.xlabel('False Positive Rate')
  pyp.ylabel('True Positive Rate')
  pyp.title('Receiver Operating Characteristic')
  pyp.legend(loc="lower right")
  pyp.grid(True)
  #pyp.show()
  pyp.savefig(results_folder+"/FinalTrainTestROCs.png")


  mela_sb = {}
  mela_seff = {}
  mela_ep = {}
  mela_cuts = {}
  dnn_sb = {}
  dnn_seff = {}
  dnn_ep = {}
  dnn_cuts = {}
  dicts = [mela_sb, mela_seff, mela_ep, mela_cuts, dnn_sb, dnn_seff, dnn_ep, dnn_cuts]
  for idict in dicts:
    idict['train'] = []
    idict['test'] = []

  full_mela = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  full_dnn = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  fweights = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}


  for cut in range(100):
    icut = cut/100.
    
    mela = {'train':{'sig':0, 'bkg':0}, 'test':{'sig':0, 'bkg':0}}
    dnn = {'train':{'sig':0, 'bkg':0}, 'test':{'sig':0, 'bkg':0}}
    for iset in Y:
        for iev in range(len(Y[iset])):
            vmela = Ymela[iset][iev]
            vdnn = Predictions[iset][iev][0]
            vweight = weights[iset][iev]
        
            #VBF (signal)
            if(Y[iset][iev] == 1):
                if(cut == 0):
                    full_mela[iset]['sig'].append( vmela )
                    full_dnn[iset]['sig'].append( vdnn )
                    fweights[iset]['sig'].append( vweight )
            
                if(vmela > icut):
                    mela[iset]['sig'] += vweight
                if(vdnn > icut):
                    dnn[iset]['sig'] += vweight
        
            #Backgrounds
            else:
                if(cut == 0):
                    full_mela[iset]['bkg'].append( vmela )
                    full_dnn[iset]['bkg'].append( vdnn )
                    fweights[iset]['bkg'].append( vweight )
            
                if(vmela > icut):
                    mela[iset]['bkg'] += vweight
                if(vdnn > icut):
                    dnn[iset]['bkg'] += vweight
    
        #computes s/sqrt(b), s_eff and s_eff*[s/(s+b)]
        if(mela[iset]['sig'] != 0 or mela[iset]['bkg'] != 0):
            mela_cuts[iset].append(icut)
            mela_sb[iset].append(mela[iset]['sig']/math.sqrt(mela[iset]['bkg']))
            mela_seff[iset].append(mela[iset]['sig']/sum(fweights[iset]['sig']))
            mela_ep[iset].append( (mela[iset]['sig']/sum(fweights[iset]['sig']))*(mela[iset]['sig']/(mela[iset]['sig']+mela[iset]['bkg'])) )        
        if(dnn[iset]['sig'] != 0 or dnn[iset]['bkg'] != 0):
            dnn_cuts[iset].append(icut)
            dnn_sb[iset].append(dnn[iset]['sig']/math.sqrt(dnn[iset]['bkg']))
            dnn_seff[iset].append(dnn[iset]['sig']/sum(fweights[iset]['sig']))
            dnn_ep[iset].append( (dnn[iset]['sig']/sum(fweights[iset]['sig']))*(dnn[iset]['sig']/(dnn[iset]['sig']+dnn[iset]['bkg'])) )        


  for iset in ['train', 'test']:
    mela_index = np.argmax(mela_ep[iset])
    print 'MELA('+iset+'): Max Seff*Purity = %.3f, S/sqrt(B) = %.3f, cut = %.3f' % (mela_ep[iset][mela_index], mela_sb[iset][mela_index], mela_cuts[iset][mela_index])
    outdict['mela'][iset+'ep'] = mela_ep[iset][mela_index]
    outdict['mela'][iset+'sb'] = mela_sb[iset][mela_index]
    outdict['mela'][iset+'cut'] = mela_cuts[iset][mela_index]
    
    dnn_index = np.argmax(dnn_ep[iset])
    print 'DNN('+iset+'): Max Seff*Purity = %.3f, S/sqrt(B) = %.3f, cut = %.3f' % (dnn_ep[iset][dnn_index], dnn_sb[iset][dnn_index], dnn_cuts[iset][dnn_index])
    outdict['nn'][iset+'ep'] = dnn_ep[iset][dnn_index]
    outdict['nn'][iset+'sb'] = dnn_sb[iset][dnn_index]
    outdict['nn'][iset+'cut'] = dnn_cuts[iset][dnn_index]

  #current metric we adopted
  pyp.rc("font", size=19)
  fig = pyp.figure()
  fig.set_figheight(7)
  fig.set_figwidth(14)

  fig1 = fig.add_subplot(121)
  pyp.plot(mela_seff['train'], mela_ep['train'], lw=1, color='red', label='MELA(train): %.3f'% mela_ep['train'][mela_index], linestyle='--')
  pyp.plot(mela_seff['test'], mela_ep['test'], lw=1, color='red', label='MELA(test): %.3f'% mela_ep['test'][mela_index], linestyle='-')
  pyp.plot(dnn_seff['train'], dnn_ep['train'], lw=1, color='blue', label='DNN(train): %.3f'% dnn_ep['train'][dnn_index], linestyle='--')
  pyp.plot(dnn_seff['test'], dnn_ep['test'], lw=1, color='blue', label='DNN(test): %.3f'% dnn_ep['test'][dnn_index], linestyle='-')
  pyp.xlim([0, 1])
  pyp.xlabel('Signal eff')
  #pyp.ylim([0, 1])
  pyp.ylabel('eff*purity')
  pyp.legend()
  pyp.grid(True)

  fig2 = fig.add_subplot(122)
  pyp.plot(mela_seff['train'], mela_sb['train'], lw=1, color='red', label='MELA(train): %.3f'% mela_sb['train'][mela_index], linestyle='--')
  pyp.plot(mela_seff['test'], mela_sb['test'], lw=1, color='red', label='MELA(test): %.3f'% mela_sb['test'][mela_index], linestyle='-')
  pyp.plot(dnn_seff['train'], dnn_sb['train'], lw=1, color='blue', label='DNN(train): %.3f'% dnn_sb['train'][dnn_index], linestyle='--')
  pyp.plot(dnn_seff['test'], dnn_sb['test'], lw=1, color='blue', label='DNN(test): %.3f'% dnn_sb['test'][dnn_index], linestyle='-')
  pyp.xlim([0, 1])
  pyp.xlabel('Signal eff')
  #pyp.ylim([0, 1])
  pyp.ylabel('S/sqrt(B)')
  pyp.legend()
  pyp.grid(True)

  pyp.tight_layout()
  #fig = pyp.show()
  pyp.savefig(results_folder+'/ComparisonMCsMetrics.png')

  ##### ---------------- Cross checking for overfitting ------------------ ####
  pyp.rc("font", size=16)
  fig = pyp.figure()
  fig.set_figheight(7)
  fig.set_figwidth(7)
  
  histos = {'train':{'sig':ROOT.TH1D('1','',40,0,1),'bkg':ROOT.TH1D('2','',40,0,1)}, 'test':{'sig':ROOT.TH1D('3','',40,0,1),'bkg':ROOT.TH1D('4','',40,0,1)}}
  centers = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  counts = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  errors = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  
  #mela plots
  for iset in ['train', 'test']:
    for imc in ['sig','bkg']:
        histos[iset][imc].Reset()
        histos[iset][imc].Sumw2()
        for i in range(len(full_mela[iset][imc])):
            histos[iset][imc].Fill(full_mela[iset][imc][i], fweights[iset][imc][i])
        histos[iset][imc].Scale(1./histos[iset][imc].Integral(), 'width')
        for ib in range(histos[iset][imc].GetNbinsX()):
            centers[iset][imc].append( histos[iset][imc].GetBinCenter(ib+1) )
            counts[iset][imc].append( histos[iset][imc].GetBinContent(ib+1) )
            errors[iset][imc].append( histos[iset][imc].GetBinError(ib+1) )

  pyp.errorbar(centers['train']['sig'], counts['train']['sig'], yerr=errors['train']['sig'], fmt='bo', markersize=1, lw=1)
  pyp.hist(centers['train']['sig'], np.linspace(0, 1, 41), weights=counts['train']['sig'], histtype='stepfilled', alpha=0.3, color='b', label = 'sig-train')
  pyp.errorbar(centers['train']['bkg'], counts['train']['bkg'], yerr=errors['train']['bkg'], fmt='ro', markersize=1, lw=1)
  pyp.hist(centers['train']['bkg'], np.linspace(0, 1, 41), weights=counts['train']['bkg'], histtype='stepfilled', alpha=0.3, color='r', label = 'bkg-train')
  pyp.errorbar(centers['test']['sig'], counts['test']['sig'], yerr=errors['test']['sig'], fmt='bo', markersize=4, lw=1, label = 'sig-test')
  #pyp.hist(centers['test']['sig'], np.linspace(0, 1, 41), weights=counts['test']['sig'], histtype='step', color='b', lw=1)
  pyp.errorbar(centers['test']['bkg'], counts['test']['bkg'], yerr=errors['test']['bkg'], fmt='ro', markersize=4, lw=1, label = 'bkg-test')
  #pyp.hist(centers['test']['bkg'], np.linspace(0, 1, 41), weights=counts['test']['bkg'], histtype='step', color='r', lw=1)
  pyp.ylabel('(a. u.)')
  pyp.xlim([-0.01,1.01])
  pyp.legend()
  pyp.xlabel('MELA')
  pyp.savefig(results_folder+'/MELAOvertrainingCheck.png')
  pyp.gca().set_yscale("log")
  pyp.savefig(results_folder+'/MELAOvertrainingCheck_logy.png')


  #--------------- DNN plots ----------------------
  pyp.rc("font", size=16)
  fig = pyp.figure()
  fig.set_figheight(7)
  fig.set_figwidth(7)

  centers = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  counts = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}
  errors = {'train':{'sig':[],'bkg':[]}, 'test':{'sig':[],'bkg':[]}}

  for iset in ['train', 'test']:
    for imc in ['sig','bkg']:
        histos[iset][imc].Reset()
        for i in range(len(full_dnn[iset][imc])):
            histos[iset][imc].Fill(full_dnn[iset][imc][i], fweights[iset][imc][i])
        histos[iset][imc].Scale(1./histos[iset][imc].Integral(), 'width')
        for ib in range(histos[iset][imc].GetNbinsX()):
            centers[iset][imc].append( histos[iset][imc].GetBinCenter(ib+1) )
            counts[iset][imc].append( histos[iset][imc].GetBinContent(ib+1) )
            errors[iset][imc].append( histos[iset][imc].GetBinError(ib+1) )

  pyp.errorbar(centers['train']['sig'], counts['train']['sig'], yerr=errors['train']['sig'], fmt='bo', markersize=1, lw=1)
  pyp.hist(centers['train']['sig'], np.linspace(0, 1, 41), weights=counts['train']['sig'], histtype='stepfilled', alpha=0.3, color='b', label = 'sig-train')
  pyp.errorbar(centers['train']['bkg'], counts['train']['bkg'], yerr=errors['train']['bkg'], fmt='ro', markersize=1, lw=1)
  pyp.hist(centers['train']['bkg'], np.linspace(0, 1, 41), weights=counts['train']['bkg'], histtype='stepfilled', alpha=0.3, color='r', label = 'bkg-train')
  pyp.errorbar(centers['test']['sig'], counts['test']['sig'], yerr=errors['test']['sig'], fmt='bo', markersize=4, lw=1, label = 'sig-test')
  #pyp.hist(centers['test']['sig'], np.linspace(0, 1, 41), weights=counts['test']['sig'], histtype='step', color='b', lw=1)
  pyp.errorbar(centers['test']['bkg'], counts['test']['bkg'], yerr=errors['test']['bkg'], fmt='ro', markersize=4, lw=1, label = 'bkg-test')
  #pyp.hist(centers['test']['bkg'], np.linspace(0, 1, 41), weights=counts['test']['bkg'], histtype='step', color='r', lw=1)
  pyp.ylabel('(a. u.)')
  pyp.xlim([-0.01,1.01])
  pyp.legend()
  pyp.xlabel('NN')
  pyp.savefig(results_folder+'/NNOvertrainingCheck.png')
  pyp.gca().set_yscale("log")
  pyp.savefig(results_folder+'/NNOvertrainingCheck_logy.png')


  #creates normalized ROCs separated by MC
  print '******** ROCs from DNN *********'
  fig = pyp.figure(figsize=(10,10))
  colors = ['red','blue','violet','green','orange','yellow','cyan','brown','gray','black']
  
  Y_truth = {}
  Y_mela = {}
  X = {}
  Weights = {}
  for ik in use_mcs:
    Y_truth[ik] = []
    X[ik] = []
    Weights[ik] = []
    Y_mela[ik] = []

    for iev in range(len(events[ik]['mc'])):
        variables = []
        for ivar in use_vars:
            variables.append(events[ik][ivar][iev])
        X[ik].append(variables)
        if(ik in signal):
            Y_truth[ik].append(1)
        else:
            Y_truth[ik].append(0)
        Weights[ik].append(events[ik]['f_weight'][iev])
        Y_mela[ik].append(events[ik]['f_Djet_VAJHU'][iev])
                
  ic = 0
  for isig in signal:
    for ik in Y_truth:
      if(ik in signal):
	continue
      cX = np.concatenate([X[isig],X[ik]])
      cY = np.concatenate([Y_truth[isig],Y_truth[ik]])
      cW = np.concatenate([Weights[isig],Weights[ik]])
      Y_score = model.predict(cX)
      fpr, tpr, thresholds = roc_curve(cY, Y_score, sample_weight=cW)
      roc_auc = auc(fpr, tpr,reorder=True)
      outdict['nn']['fullroc_'+isig+ik] = roc_auc
      print '(DNN)  %s vs %s -- ROC AUC: %.2f' % (isig, ik, roc_auc)
      pyp.plot(fpr, tpr, lw=1, color=colors[ic], label=('%s vs %s (%0.2f)' % (isig,ik,roc_auc)))
      dY = np.concatenate([Y_mela[isig],Y_mela[ik]])
      fpr, tpr, thresholds = roc_curve(cY, dY, sample_weight=cW)
      roc_auc = auc(fpr, tpr,reorder=True)
      outdict['mela']['fullroc_'+isig+ik] = roc_auc
      print '(MELA) %s vs %s -- ROC AUC: %.2f' % (isig, ik, roc_auc)
      pyp.plot(fpr, tpr, linestyle='--', lw=1, color=colors[ic], label=('%s vs %s (%0.2f)' % (isig,ik,roc_auc)))
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
  #pyp.show()
  pyp.savefig(results_folder+'/ComparisonMCsROC.png')
  
  
  print 'Saving dictionary with summary of results...'
  fileout = open(results_folder+'/SummaryOfResults.pkl','w')
  pickle.dump( outdict, fileout )
  fileout.close()
###--------------------------------------------------------------------------------------------------------------###



def main(options):
  #Setting defeult options in case omitted
  if(options.resultsfolder == None):
    options.resultsfolder = 'results'
  if(options.nninputs == None):
    raise ValueError('Variables to be used not informed! Please state them!')
  if(options.signal == None):
    raise ValueError('Signal key(s) not specified. Please state them!')
  if(options.preproc == None):
    options.preproc = ['none']
  if(options.topology == None):
    options.topology = [10]  
  if(options.neuron == None):
    options.neuron = ['relu']
  if(options.minimizer == None):
    options.minimizer = ['adam']
  if(options.scaletrain == None):
    options.scaletrain = ['none']
  if(options.nooutliers == None):
    options.nooutliers = False

  print '----- CONFIGURATION TO BE USED ------'
  print 'infile: ', options.infile
  print 'resultsfolder: ', options.resultsfolder
  print 'keys: ', options.keys
  print 'signal: ', options.signal
  print 'nninputs: ', options.nninputs
  print 'split: ', options.split  
  print 'preproc: ', options.preproc
  print 'topology: ', options.topology
  print 'neuron: ', options.neuron
  print 'nepochs: ', options.nepochs
  print 'patience: ', options.patience
  print 'batchsize: ', options.batchsize
  print 'minimizer: ', options.minimizer
  print 'scaletrain: ', options.scaletrain
  print 'nooutliers: ', options.nooutliers

  print '----- NN TRAINING STARTING ------'
  swatch = ROOT.TStopwatch()
  swatch.Start()
  
  TrainNeuralNetwork(options.infile,
		     options.resultsfolder,
		     options.keys,
		     options.signal,
		     options.nninputs,
		     options.split,
		     options.preproc,
		     options.topology,
		     options.neuron,
		     options.nepochs,
		     options.patience,
		     options.batchsize,
		     options.minimizer,
		     options.scaletrain,
                     options.nooutliers)
  
  print '----- NN TRAINING FINISHED ------'
  print 'Time %.1f (seconds)' % swatch.RealTime()
  
  
if __name__ == '__main__':

 # Setup argument parser
 parser = argparse.ArgumentParser()

 # Add more arguments
 parser.add_argument("--infile", help="Name of dictionary from FormatROOTs.py containing the formated variables from root files")
 parser.add_argument("--resultsfolder", help="Name of folder to store the results")
 parser.add_argument("--keys", nargs='+', help="Name of the MCs to be used")
 parser.add_argument("--signal", nargs='+', help="Name of the MCs to be used as signal (the remaining will be taken as background)")
 parser.add_argument("--nninputs", nargs='+', help="Variables to be used as inputs for the NN")
 parser.add_argument("--split", type=float, default=0.5, help="Fraction of each MC to be used for train (the remaining is used for test)")
 parser.add_argument("--preproc", action="append", help="Pre-processing of inputs: none, normalize or scale")
 parser.add_argument("--topology", type=int, nargs='+', help="The topology of the NN. Ex: 21:13:8, a NN with 3 hidden layers")
 parser.add_argument("--neuron", action="append", help="The type of neuron to be used: relu, sigmod or tanh")
 parser.add_argument("--nepochs", type=int, default=1000, help="Number of epochs to train")
 parser.add_argument("--patience", type=int, default=100, help="Number of epochs to wait without improvement before stop training")
 parser.add_argument("--batchsize", type=int, default=32, help="Size of the batch to be used in each update")
 parser.add_argument("--minimizer", action="append", help="Minimizer: sgd, adam, adagrad, adadelta, rmsprop")
 parser.add_argument("--scaletrain", action="append", help="Type of scaling to be used into the training: none, mc_weight (XS) or event_weight (individual weight)")
 parser.add_argument("--nooutliers", action="store_true", help="When this flag is set outlier events are not used in NN analysis")

 # Parse default arguments
 options = parser.parse_args()
 main(options)
    
