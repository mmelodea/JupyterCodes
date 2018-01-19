import os
import sys
import cPickle as pickle
from UserFunctions import formatInputs

#loads input data
path = '/home/micah/temp/hzz4l_histos/OrganizedSamples/'
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

#format inputs (Data & MC)
comparison = '>='
njets = 2
min4lmass = 118
max4lmass = 130
events = formatInputs(files_addresses, comparison, njets, min4lmass, max4lmass)

keys = []
for ik in events:
    keys.append(ik)
print '----------- KEYS -----------'
print keys    

#for randomizing FS composition in samples
#print 'shuffling events...'
import random
random.seed(999)
for ik in events:
    random.shuffle(events[ik])

#save the dictionary
fileout = open('hzz4l_{0}jets_m4l{1}-{2}GeV_shuffled2e2mu.pkl'.format(njets,min4lmass,max4lmass),'w')
pickle.dump( events, fileout )
fileout.close()
