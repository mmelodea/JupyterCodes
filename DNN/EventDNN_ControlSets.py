
# coding: utf-8

# In[1]:


#set adequate environment
import os
import sys
import theano
import keras
theano.config.gcc.cxxflags = '-march=corei7'


# In[2]:


#load needed things
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop, Nadam, AMSgrad
from keras.layers import Input, Activation, Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
# Run classifier with cross-validation and plot ROC curves
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import interp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp
import itertools
import math
import ROOT
import cPickle as pickle
from UserFunctions import *

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[3]:


#load events
print "Loading events..."
comparison = '>='
njets = 2
min4lmass = 118
max4lmass = 130

filein = open('hzz4l_{0}jets_m4l{1}-{2}GeV_shuffledFS.pkl'.format(njets,min4lmass,max4lmass),'r')
events = pickle.load( filein )
filein.close()


# In[4]:


del events['myqqZZ']
del events['ggH']

#shows events and statistics
class_weight = {}
keys = []
for ik in events:
    nevents = 0
    sumw = 0
    nevents = len(events[ik])
    for i in range(len(events[ik])):
        sumw += events[ik][i][2]

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


#to select how many jets to use;  eg. 7 = 4leptons + 3jets
nparticles = 7
features = {
    'pt' : None,
    'eta': None,
    'phi': None
    #'e'  : None
}
nfeatures = len(features)


# In[9]:


#just to organize index of important quantities
djet_index   = 1
mela_index   = 2
weight_index = 3
class_weight_index = 4

#prepare train set
X = {}
Y = {}
X['train'], Y['train'], Ydjet_train, Ymela_train, weights_train_set, scales_train_set = prepareSet(full_event_train, nparticles, nfeatures, djet_index, mela_index, weight_index, class_weight_index)


# In[10]:


#prepare test set
X['test'], Y['test'], Ydjet_test, Ymela_test, weights_test_set, scales_test_set = prepareSet(full_event_test, nparticles, nfeatures, djet_index, mela_index, weight_index, class_weight_index)




# In[12]:


#parameters for training
nepochs = 400
wait_for = 100
sbatch = 128
opt = Adam()
#opt = SGD(lr=0.001, momentum=0.3, nesterov=True)
#opt = Nadam()
#opt = AMSgrad()
#opt = Adadelta()

# DNN Model
from keras.layers import Dropout
from keras.layers import ELU

nn15_model = Sequential()
#shallow network
nn15_model.add(Dense(15, input_shape=(nparticles*nfeatures,), activation='relu'))
nn15_model.add(Dense(1, activation='sigmoid'))
nn15_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print nn15_model.summary()

dnn_model = Sequential()
dnn_model.add(Dense(9, input_shape=(nparticles*nfeatures,), activation='relu'))
dnn_model.add(Dense(5, activation='relu'))
dnn_model.add(Dense(3, activation='relu'))

dnn_model.add(Dense(1, activation='sigmoid'))
dnn_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#dnn_model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
print dnn_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=wait_for)
filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')


# In[13]:


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
        
        #c_lrate = K.eval(self.model.optimizer.lr)
        #changing lr based on loss
        if(self.i > 100):
	  n_lrate = 0.001 - 0.001*(self.i/float(nepochs))
          K.set_value(self.model.optimizer.lr, n_lrate)
        
        print "epoch: {0}, tloss: {1:.3f}, vloss: {2:.3f}, troc: {3:.3f}, vroc: {4:.3f}, lr: {5}".format(self.i, self.losses[len(self.losses)-1], self.val_losses[len(self.val_losses)-1], self.trocs[len(self.trocs)-1], self.vrocs[len(self.vrocs)-1], K.eval(self.model.optimizer.lr))

        
        #stop training if training and testing Loss get too away
        #if(epoch > 50):
        #    train_sum = 0
        #    val_sum = 0
        #    for i in range(30):
        #        train_sum += self.losses[len(self.losses)-1-i]/6.
        #        val_sum += self.val_losses[len(self.val_losses)-1-i]/6.
        #    if(train_sum-val_sum < -0.02):
        #        self.model.stop_training = True
        #        print "Stoping training!"
                
            
Tschedule = Tscheduler()


# In[ ]:


#train the network
#dnn_model.load_weights("weights_best2.hdf5")
history = dnn_model.fit(X['train'],
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
                        verbose=0,
                        callbacks=[checkpoint,early_stopping,Tschedule]
                       )
			
			
# load final best weights
dnn_model.load_weights("weights_best.hdf5")
			
			
fpr, tpr, thresholds = roc_curve(Y['test'], Ydjet_test, sample_weight=weights_test_set)
roc_auc = auc(fpr, tpr,reorder=True)
print 'Djet test AUC = %0.3f' % roc_auc

fpr, tpr, thresholds = roc_curve(Y['test'], Ymela_test, sample_weight=weights_test_set)
roc_auc = auc(fpr, tpr,reorder=True)
print 'MELA test AUC = %0.3f' % roc_auc

Y_score = dnn_model.predict(X['test'])
fpr, tpr, thresholds = roc_curve(Y['test'], Y_score, sample_weight=weights_test_set)
roc_auc = auc(fpr, tpr,reorder=True)
print 'DNN test AUC = %0.3f' % roc_auc
