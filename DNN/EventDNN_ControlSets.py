
# coding: utf-8

# In[1]:


#set adequate environment
import os
import sys
import keras
#theano.config.gcc.cxxflags = '-march=corei7'


# In[2]:


#load needed things
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.layers import Input, Activation, Dense
#from keras.utils import np_utils
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# Run classifier with cross-validation and plot ROC curves
#from itertools import cycle
#from sklearn.metrics import roc_curve, auc
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
#from scipy import interp
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as pyp
#import itertools
import math
#import ROOT
import cPickle as pickle
from UserFunctions import *

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[3]:


#load events
comparison = '>='
njets = 2
min4lmass = 118
max4lmass = 130

filein = open('hzz4l_{0}jets_m4l{1}-{2}GeV_shuffledFS.pkl'.format(njets,min4lmass,max4lmass),'r')
events = pickle.load( filein )
filein.close()


# In[4]:


#del events['myqqZZ']
#del events['ggH']

#shows events and statistics
class_weight = {}
keys = []
for ik in events:
    nevents = 0
    sumw = 0
    nevents = len(events[ik])
    for i in range(len(events[ik])):
        sumw += events[ik][i][0]

    print '%s events: %i, normalized: %.4f' % (ik,nevents,sumw)
    class_weight[ik] = sumw
    keys.append(ik) 
    
#only MC
summ = 0
for ik in events:
    if(ik != 'Data'):
        summ += len(events[ik])
print 'MC total events: %i' % summ
class_weight['myqqZZ'] = class_weight['qqZZ']
vhcw = class_weight['WH']+class_weight['ZH']
class_weight['WH'] = vhcw
class_weight['ZH'] = vhcw




# In[6]:


#creates a copy from orig events
orig_events = events
events = {}
events['VBF'] = orig_events['VBF']
events['HJJ'] = orig_events['HJJ']
events['ttH'] = orig_events['ttH']
events['ZH'] = orig_events['ZH']
events['WH'] = orig_events['WH']
events['qqZZ'] = orig_events['qqZZ']
events['ggZZ'] = orig_events['ggZZ']
events['ttZ'] = orig_events['ttZ']
#events['myqqZZ'] = orig_events['myqqZZ']


# In[7]:


#organize events and splits mc into train, validation and test; data it's kept full
#the split_factor sets train size then validation and test sets have the same size
split_factor = 0.8
full_event_train, full_event_test, full_event_data = splitInputs(events, split_factor, class_weight)


# In[8]:


#just to organize index of important quantities
class_weight_index = 1
weight_index = 2
djet_index   = 3
mela_index   = 4

#prepare train set
X = {}
Y = {}
X['train'], Y['train'], Ydjet_train, Ymela_train, weights_train_set, scales_train_set = prepareSet(full_event_train, djet_index, mela_index, weight_index, class_weight_index)


# In[9]:


#prepare test set
X['test'], Y['test'], Ydjet_test, Ymela_test, weights_test_set, scales_test_set = prepareSet(full_event_test, djet_index, mela_index, weight_index, class_weight_index)


# In[10]:


#parameters for training
nepochs = 20
wait_for = 60
sbatch = 32
opt = Adam()
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=wait_for)


# In[15]:


#DNN network
print "---------- DNN Topology --------- "
dnn_model = Sequential()
dnn_model.add(Dense(7, input_shape=(len(X['train'][0]),), activation='relu', kernel_initializer='random_uniform'))
dnn_model.add(Dense(5, activation='relu', kernel_initializer='random_uniform'))    
dnn_model.add(Dense(3, activation='relu', kernel_initializer='random_uniform'))    
#dnn_model.add(Dense(6, activation='relu', kernel_initializer='random_uniform'))
#dnn_model.add(Dense(4, activation='relu', kernel_initializer='random_uniform'))
dnn_model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))    
dnn_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print dnn_model.summary()




# In[ ]:


#filepath1="weights_dnn.hdf5"
#checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

#train the DNN
history1 = dnn_model.fit(X['train'],
                        Y['train'], 
                        sample_weight=scales_train_set,
                        #sample_weight=weights_train_set,
                        validation_data=(X['test'],
                                         Y['test']
                                         ,scales_test_set
                                         #,weights_test_set
                                        ), 
                        epochs=nepochs,
                        batch_size=sbatch, 
                        verbose=2
                        #,callbacks=[checkpoint1,early_stopping]
                       )


# In[ ]:


# load final best weights
#dnn_model.load_weights(filepath1)



#train set normalized roc
#Y_score = dnn_model.predict(X['train'])
#fpr, tpr, thresholds = roc_curve(Y['train'], Y_score, sample_weight=weights_train_set)
#roc_auc = auc(fpr, tpr,reorder=True)
#print "DNN train AUC = %0.3f" % roc_auc

#test set unormalized roc
#Y_score = dnn_model.predict(X['test'])
#fpr, tpr, thresholds = roc_curve(Y['test'], Y_score, sample_weight=weights_test_set)
#roc_auc = auc(fpr, tpr,reorder=True)
#print "DNN test AUC = %0.3f" % roc_auc
