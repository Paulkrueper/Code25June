#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In this example we will train very simple NNs to differentiate between a CP-even and a CP-odd Higgs for the rhorho channel
# One BDT will use only 1 variable analogous to current methodology, the second BDT will include additional information
# to help improve the seperation 


# In[2]:



#!pip install --user uproot
import sys
sys.path.append("/eos/home-m/dwinterb/.local/lib/python2.7/site-packages")


# In[3]:
import os
import json
import uproot 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from keras.callbacks import History 


# In[4]:


# loading the tree

#path1="/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/SUSYGluGluToBBHToTauTau_M-2600_powheg_tt_2018.root"
#path2="/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY1JetsToLL-LO_tt_2018.root"


eventlist=["/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/SUSYGluGluToBBHToTauTau_M-2600_powheg_tt_2018.root"] #tbd
bcglist=["/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY1JetsToLL-LO_tt_2018.root","/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DYJetsToLL-LO_tt_2018.root"] #tbd
nameslist=["DY1JetsToLL-LO","DYJetsToLL-LO"]

backgroundlist=[]
for i in bcglist:
	split=os.path.split(i)
	tail=split[1]
	backgroundlist.append(tail)


weightlist=json.load(open("/vols/cms/pfk18/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/scripts/params_mssm_2018.json"))

lumi=59740 #for Tau

Weights=[]
for bcg in nameslist:
	Dict=weightlist[bcg]
	xs=Dict["xs"]
	evt=Dict["evt"]
	weight=xs*lumi/evt
	Weights.append(weight)






# In[5]:


# define what variables are to be read into the dataframe
def MAIN(background,weight_normalisation):

	variables = [  
                "wt", 
		"pt_1","pt_2",
                "met",
                "deepTauVsJets_medium_1","deepTauVsJets_medium_2",
                "deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
                "deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
                "trg_doubletau",
             ]
	

	tree =uproot.open(eventlist[0])["ntuple"]
	tree2 =uproot.open(background)["ntuple"]



	df = tree.pandas.df(variables)

	df2 = tree2.pandas.df(variables)


# In[6]:


# apply some preselections, these selections are used to mimic those used in the analysis and to select only rhorho events
# also use random number "rand" and tau spinner weights "wt_cp_{sm,ps,mm}" to select a sample of CP-even and CP-odd
# like events. the weights go beween 0 and 2 so by dividing by 2 we can interpret these as probabilities and select
# CP-even(odd) events if the rand is less than this probability 


	import random
	random.seed(123456)

	df_ps = df[
    # comment some selections to help with stats
 	   (df["deepTauVsJets_medium_1"] > 0.5) 
 	   & (df["deepTauVsEle_vvloose_1"] > 0.5)
    	   & (df["deepTauVsMu_vloose_1"] > 0.5)
	   & (df["deepTauVsJets_medium_2"] > 0.5) 
	    & (df["deepTauVsEle_vvloose_2"] > 0.5)
	    & (df["deepTauVsMu_vloose_2"] > 0.5)
 	   & (df["trg_doubletau"] > 0.5)
	]

	df_sm = df2[
    # comment some selections to help with stats
    	(df2["deepTauVsJets_medium_1"] > 0.5)
    	& (df2["deepTauVsEle_vvloose_1"] > 0.5)
    	& (df2["deepTauVsMu_vloose_1"] > 0.5)
    	& (df2["deepTauVsJets_medium_2"] > 0.5)
    	& (df2["deepTauVsEle_vvloose_2"] > 0.5)
   	& (df2["deepTauVsMu_vloose_2"] > 0.5)
    	& (df2["trg_doubletau"] > 0.5)]

# In[7]:


# create target labels (y)
         
# prepare the target labels
	y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
	y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))

	y = pd.concat([y_sm, y_ps])
	y.columns = ["class"]

# prepare the dataframe to use in training
	X = pd.concat([df_sm, df_ps])



# drop any other variables that aren't required in training

	X = X.drop([
          
            "deepTauVsJets_medium_1","deepTauVsJets_medium_2",
            "deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
            "deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
            "trg_doubletau",
           ], axis=1).reset_index(drop=True) 


# In[8]:


# define function to plot 'signal' vs 'background' for a specified variables
# useful to check whether a variable gives some separation between
# signal and background states
	def plot_signal_background(data1, data2, column,
                        bins=100, x_uplim=0, **kwargs):

	    	if "alpha" not in kwargs:
       			kwargs["alpha"] = 0.5

    			df1 = data1[column]
    			df2 = data2[column]

    			fig, ax = plt.subplots()
    			df1 = df1.sample(3000, random_state=1234)
    			df2 = df2.sample(3000, random_state=1234)
    			low = max(min(df1.min(), df2.min()),-5)
    			high = max(df1.max(), df2.max())
    		if x_uplim != 0: 
			high = x_uplim

    			ax.hist(df1, bins=bins, range=(low,high), **kwargs)
    			ax.hist(df2, bins=bins, range=(low,high), **kwargs)
    			
    		if x_uplim != 0:
        		ax.set_xlim(0,x_uplim)

    # ax.set_yscale('log')


# In[9]:


# make plots of all variables

	for key, values in X.iteritems():
	    print(key)
    	print("A")
    	print(values)
    	plot_signal_background(df_sm, df_ps, key, bins=100)


# In[10]:


# recale variables so that they go between 0-1 
# this is improtant for neural networks - see https://www.jeremyjordan.me/batch-normalization/ for details

	scaler_x = MinMaxScaler()
	scaler_y = MinMaxScaler()
	print(scaler_x.fit(X))
	xscale=scaler_x.transform(X)

	X = pd.DataFrame(xscale,columns=X.columns)


# In[11]:


# split X1, X2, and y into train and validation dataset 

	X_train,X_test, y_train, y_test  = train_test_split(
	    X,
	    y,
	    test_size=0.2,
	    random_state=123456,
	    stratify=y.values,
	)	


# In[12]:


# define a simple NN
	def baseline_model():
    # create model
	    model = Sequential()
	    model.add(Dense(len(X.columns), input_dim=len(X.columns), kernel_initializer='normal', activation='relu'))
	    model.add(Dense((len(X.columns))*2, kernel_initializer='normal', activation='relu'))
	    model.add(Dense(1, activation="sigmoid"))
	    model.compile(loss='binary_crossentropy', optimizer='adam')  
	    return model


# In[13]:


# define early stopping
	early_stop = EarlyStopping(monitor='val_loss',patience=10)


# In[19]:


# first run the training for simple case with just 1 variable
	history = History()

	model = baseline_model()

	model.fit(
                X_train, y_train,
                #sample_weight=w_train,
                batch_size=10000,
                epochs=100,
                callbacks=[history,early_stop],
                validation_data=(X_test, y_test))#, w_val))


# In[15]:


# Extract number of run epochs from the training history
	epochs = range(1, len(history.history["loss"])+1)

# Extract loss on training and validation ddataset and plot them together
	plt.plot(epochs, history.history["loss"], "o-", label="Training")
	plt.plot(epochs, history.history["val_loss"], "o-", label="Test")
	plt.xlabel("Epochs"), plt.ylabel("Loss")
	plt.yscale("log")
	plt.legend();

# differencwes between the loss for training vs test implies overtraining


# In[16]:


	prediction = model.predict(X_test)


# In[17]:


#  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
	def plot_roc_curve(fpr, tpr, auc):
    		fig, ax = plt.subplots()
    		ax.plot(fpr, tpr)
    		ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    		ax.grid()
    		ax.text(0.6, 0.3, 'ROC AUC Score: {:.3f}'.format(auc),
    	        bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    		lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    		ax.plot(lims, lims, 'k--')
    		ax.set_xlim(lims)
    		ax.set_ylim(lims)
    		plt.savefig('roc_rho_rho_NN')


# In[18]:


# plot ROC curve for improved training
	y_proba = model.predict_proba(X_test) # outputs two probabilties
	auc = roc_auc_score(y_test, y_proba)
	fpr, tpr, _ = roc_curve(y_test, y_proba)
	plot_roc_curve(fpr, tpr, auc)
#	return wt*weight_normalisation*auc
	return weight_normalisation*auc
List=[]
index=0
for i in bcglist:
	w=Weights[index]
	weightedAUC=MAIN(i,w)
	List.append(weightedAUC)
	index+=1
j=0
for i in List:
	j+=i
print(j)
	
	
