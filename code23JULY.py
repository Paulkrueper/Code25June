

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
import matplotlib.pyplot as plt

# In[4]:


# loading the tree

eventlist=["/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/SUSYGluGluToBBHToTauTau_M-2600_powheg_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU2_gU3_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU2_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU3_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU2_gU3_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU3_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU3_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU3_gU3_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU3_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU4_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU3_gU3_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU4_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU4_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU2_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU4_gU3_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU4_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_0_mU2_gU2_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU2_gU1_tt_2018.root",
"/vols/cms/gu18/Offline/output/MSSM/vlq_2018_v3/VectorLQToTauTau_betaRd33_minus1_mU4_gU3_tt_2018.root",
] 


bcglist=["/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DYJetsToLL-LO_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY1JetsToLL-LO_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY2JetsToLL-LO_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY3JetsToLL-LO_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/DY4JetsToLL-LO_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/EWKZ2Jets_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/TTTo2L2Nu_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/TTToHadronic_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/TTToSemiLeptonic_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/T-tW-ext1_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/Tbar-tW-ext1_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/Tbar-t_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WWTo2L2Nu_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/T-t_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WWToLNuQQ_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WZTo1L3Nu_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WZTo3LNu_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WZTo3LNu-ext1_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/WZTo2L2Q_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/ZZTo2L2Nu-ext1_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/ZZTo2L2Nu-ext2_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/ZZTo2L2Q_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/ZZTo4L-ext_tt_2018.root",
"/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/ZZTo4L_tt_2018.root"

] #tbd
nameslist=["DYJetsToLL-LO","DY1JetsToLL-LO","DY2JetsToLL-LO","DY3JetsToLL-LO","DY4JetsToLL-LO","EWKZ2Jets","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic","T-tW-ext1", "Tbar-tW-ext1","Tbar-t","WWTo2L2Nu","T-t",
          "WWToLNuQQ","WZTo1L3Nu","WZTo3LNu","WZTo3LNu-ext1","WZTo2L2Q",
          "ZZTo2L2Nu-ext1","ZZTo2L2Nu-ext2","ZZTo2L2Q","ZZTo4L-ext","ZZTo4L"
]



weightlist=json.load(open("/vols/cms/pfk18/CMSSW_10_2_19/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/scripts/params_mssm_2018.json"))

lumi=weightlist["Tau"]["lumi"] #for Tau


def dataframe(bcg,name):
	if name != 0:
		Dict=weightlist[name]
		xs=Dict["xs"]
		evt=Dict["evt"]
		weight=xs*lumi/evt
	else:
		weight = 1



# In[5]:


# define what variables are to be read into the dataframe
	variables = [  
                "wt", 
		"pt_1","pt_2",
                "met",
                "deepTauVsJets_medium_1","deepTauVsJets_medium_2",
                "deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
                "deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
                "trg_doubletau",
#		"deepTauVsJets_vvvloose_1",
#		"deepTauVsJets_medium_1"             ]
]		
	tree =uproot.open(bcg)["ntuple"]

	df = tree.pandas.df(variables)
	print("df")
#	print(df)
# In[6]:


# apply some preselections, these selections are used to mimic those used in the analysis and to select only rhorho events
# also use random number "rand" and tau spinner weights "wt_cp_{sm,ps,mm}" to select a sample of CP-even and CP-odd
# like events. the weights go beween 0 and 2 so by dividing by 2 we can interpret these as probabilities and select
# CP-even(odd) events if the rand is less than this probability 


	import random
	random.seed(123456)

	df_sv = df[
    # comment some selections to help with stats
 	   (df["deepTauVsJets_medium_1"] > 0.5) 
 	   & (df["deepTauVsEle_vvloose_1"] > 0.5)
    	   & (df["deepTauVsMu_vloose_1"] > 0.5)
	   & (df["deepTauVsJets_medium_2"] > 0.5) 
	    & (df["deepTauVsEle_vvloose_2"] > 0.5)
	    & (df["deepTauVsMu_vloose_2"] > 0.5)
 	   & (df["trg_doubletau"] > 0.5)
#	   & (df["deepTauVsJets_vvvloose_1"]>0.5)
#	   & (df["deepTauVsJets_medium_1"]<0.5)
	]
	
	print("A")
	print(type(df_sv))
	print("B")
	print(np.shape(df_sv))
	print(np.shape(df_sv)[0])
	print("C")
	array=np.full(np.shape(df_sv)[0],weight)
#	List=array.tolist(array)
	print(array)
	print(np.shape(array))
	df_sv.insert(1,"wt2",array,True)
	print("D")
	print(np.shape(df_sv))
	print(df_sv)
	print("E")
	print(df_sv["wt2"])
	print("wt")

	#Alternative
	#address_weight=np.full(np.shape(df_sv)[0],weight)
#	df_sv["wt"]=address_weight



	#Alternative2
#	df_sv = df_sv.assign(weight = np.full(np.shape(df_sv)[0],weight)


	return  df_sv

# In[7]:
df_svlist=[]
i=0
for bcg in bcglist:
	name=nameslist[i]
	df_sv=dataframe(bcg,name)
	df_svlist.append(df_sv)	
	i+=1

dfsignallist=[]
for i in eventlist:
	r=dataframe(i,0)
        df=r
        dfsignallist.append(df)
#signal=dataframe("/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2/SUSYGluGluToBBHToTauTau_M-2600_powheg_tt_2018.root",0)



# prepare the dataframe to use in training
X_bcg = pd.concat(df_svlist)
X_sig=pd.concat(dfsignallist)

X=pd.concat([X_bcg,X_sig])

	
wt = X["wt"]*X["wt2"]
print("weight5")
print(wt[:5])
#weight_normalisation=dataframe(bcglist[0],nameslist[0])[2]
#wt = wt*weight_normalisation
print(wt[:5])
# drop any other variables that aren't required in training

X = X.drop([
"wt",
"wt2",
"deepTauVsJets_medium_1","deepTauVsJets_medium_2",
"deepTauVsEle_vvloose_1","deepTauVsEle_vvloose_2",
"deepTauVsMu_vloose_1","deepTauVsMu_vloose_2",
#"trg_doubletau","deepTauVsJets_vvvloose_1",
#"deepTauVsJets_medium_1"
], axis=1).reset_index(drop=True)
print(X)



#target labels
ybcglist=[]
i=0
for item  in df_svlist:
        print("index",df_svlist[i])
        print("shape",df_svlist[i].shape[0])
        y_bcg=pd.DataFrame(np.ones(df_svlist[i].shape[0]))
        ybcglist.append(y_bcg)
        i+=1
ysiglist=[]
j=0
for item in dfsignallist:
	y_sig=pd.DataFrame(np.zeros(dfsignallist[j].shape[0]))
	ysiglist.append(y_sig)	
	j+=1

ybg=pd.concat(ybcglist)
ysig=pd.concat(ysiglist)
y=pd.concat([ybg,ysig])
print("yvalues")
print(y)
print(y.values)
columns=["class"]



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
    	plot_signal_background(df_svlist[0],dfsignallist[0],key,bins=100)


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
print("Y")
#for i in y.values:

#	print(i)
print("HERE")
print(np.shape(X))
print(X)
print("HERE2")
#iith pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
print(np.shape(y))
print(y)


print(len(X))
print(len(y))
print(len(wt))

X_train,X_test, y_train, y_test,wt_train,wt_test = train_test_split(
	X,
	y,
	wt,
	test_size=0.2,
	random_state=123456,
	stratify=y[0],
	)	


# In[12]:

print(X.columns)
print(len(X.columns))
print(np.shape(X))
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
print(np.shape(wt_test))
print(np.shape(X_test))
print(np.shape(y_test))
print(np.shape(wt_train))
print(np.shape(X_train))
print(np.shape(y_train))


model.fit(X_train, y_train,sample_weight=wt_train,epochs=1,callbacks=[history,early_stop],steps_per_epoch=200,validation_steps=180,validation_data=(X_test, y_test,wt_test))


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
print(i)
print(y_test)
auc = roc_auc_score(y_test, y_proba)
print(auc)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plot_roc_curve(tpr, fpr,auc)
#plt.plot(y_proba[0],y_proba[1],"o")