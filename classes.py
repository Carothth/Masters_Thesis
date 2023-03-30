# -*- coding: utf-8 -*-
"""
Created on Aug 13 2022
Last edited on Mar 22 2023
@author: carol

This file is developed for the master's thesis "Clustering Based Explanations
of Decisions Structures in Deep Neural Networks". It contains classes for 
handeling tasks related to inspection of feed forward neural networks(FNN's).
                        
The first class 'layer_outputs' contains one function, returning the 
intermediate outputs of a FNN, i.e. the output of each layer of the network
after training.

The second class 'CustomCallback' contains a function returning a customized
callback to be used in the training of the network if one want to have the
data from each layer after each epoch during training.

The third class 'clustering_layers' contains a function returning a clustering 
of each layer in a given FNN. Different clustering algorithms can be chosen.

The fourth class 'layer_plots' contains two functions. The first 'pc_plots' 
constructs a figure with parallel coordinate plots of each of the layers after
training og the FNN. The second 'pcplot_cluster_layers' constructs a figure 
with parallel coordinate plots for each cluster of each layer after training 
of the FNN.

The fifth class 'epoch_plots' contains one function 'pcplot' returning a list
of figure objects. Each of them is a figure with parallel coordinate plots of
each of the layers efter each epoch during training of the FNN.

The sixth class 'cluster_measures' contains five functions. The first two 
'mean_layer_meas' and 'mean_layer_meas_no_out' computing the mean of a 
similarity measure across the layers, with and without the output layer 
respectively. The last three functions 'moving_block_mean', 
'moving_block_meansd' and 'moving_block_sdmean' finds an estimate of the 
optimal epoch according to a moving block approach. The three functions finds
the optimal epoch by using only the mean, using first the mean and then the 
standard deviation and first the standard deviation and then the mean 
repectively.

"""

import os 
path = "C:/Users/carol/OneDrive/10semester/Speciale/handin"
os.chdir(path)

from keras import Model
from sklearn.cluster import KMeans
import hdbscan
from sklearn.cluster import AgglomerativeClustering 

#EM-clustering/Gaussian Mixture model
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
from sklearn.mixture import GaussianMixture

#https://github.com/ChongYou/subspace-clustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

import pandas as pd
import numpy as np

from matplotlib.pyplot import cm
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt 

#import tensorflow as tf
from keras.callbacks import Callback
import pickle


class layer_outputs():
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def getIntermediateOutputs(self):
        intermediate_output = [[]]*len(self.model.layers)
        dfs = []
        for i in range(len(self.model.layers)):
            layer_name = "layer_"+str(i)
            intermediate_layer_model = Model(inputs=self.model.input,outputs=self.model.get_layer(layer_name).output)
            intermediate_output[i] = intermediate_layer_model.predict(self.X_test)
            
            if len(intermediate_output[i].shape)==2:
                #Reshaped intermediate outputs 
                rio = intermediate_output[i]
                dfs.append(rio)
            elif len(intermediate_output[i].shape)==3: 
                rio = intermediate_output[i].reshape(len(intermediate_output[i]),len(intermediate_output[i][0])*len(intermediate_output[i][0][0]))
                dfs.append(rio)
            elif len(intermediate_output[i].shape)==4: 
                if intermediate_output[i].shape[3]==1:
                    rio = intermediate_output[i].reshape(len(intermediate_output[i]),len(intermediate_output[i][0])*len(intermediate_output[i][0][0])**len(intermediate_output[i][0][0][0]))
                    dfs.append(rio)
                else:
                    print("not implemented yet")
            else:
                print("error")
        return dfs, intermediate_output


#Custom callback that saves output of each layer in each epoch as pikle objects
#Doc: https://keras.io/guides/writing_your_own_callbacks/
#https://stackabuse.com/custom-keras-callback-for-saving-prediction-on-each-epoch-with-visualizations/
#Doc on pickle: https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence

#Get predictions after each epoch:
#https://stackoverflow.com/questions/36864774/python-keras-how-to-access-each-epoch-prediction

class CustomCallback(Callback):
    def __init__(self, X_test, y_test, model, path):
        #super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
        filename = 'df'
        outputss = layer_outputs(self.model, self.X_test)
        dfs, intermediate_output = outputss.getIntermediateOutputs()
        #Save output of each layer in each epoch as pickle objects
        with open(self.path+'/{}_{}.pkl'.format(filename, epoch), 'wb') as out_file:
            pickle.dump(dfs, out_file, pickle.HIGHEST_PROTOCOL)
        
        #Save test accuracy and loss, evaluated in each epoch, as pickle objects
        testeval = self.model.evaluate(self.X_test, self.y_test)
        filename2 = 'test_eval'
        with open(self.path+'/{}_{}.pkl'.format(filename2, epoch), 'wb') as out_file2:
            pickle.dump(testeval, out_file2, pickle.HIGHEST_PROTOCOL)
        
        
class clustering_layers():
    def __init__(self, method, model, dfs, n_clusters, X_test, org_labs, random_state=None):
        self.dfs = dfs
        self.model = model
        self.n_clusters = n_clusters
        self.X_test = X_test
        self.org_labs = org_labs
        self.random_state = random_state
        self.method = method
        
        if len(self.X_test.shape)==4: 
            if self.X_test.shape[3]==1:
                self.X_test = self.X_test.reshape(len(self.X_test),len(self.X_test[0])*len(self.X_test[0][0])**len(self.X_test[0][0][0]))

    def get_clusters(self):
        #Initialize dependent on the clustering algorithm
        if self.method == "kmeans":
            clus = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.method == "hdbscan":
            clus = hdbscan.HDBSCAN()
        elif self.method == "agglomerative":
            clus = AgglomerativeClustering(self.n_clusters, affinity = 'euclidean', linkage = 'ward')
        elif self.method == "gauss":
           clus = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        else:
            print("Choose one of following clustering methods: kmeans, hdbscan, agglomerative, gauss")
        
        #Gaussian Mixture handles the labels differently
        clus_org = clus.fit(self.X_test)
        if self.method=="gauss":
            clus_org_labels = pd.DataFrame(clus.predict(self.X_test))
        else:
            clus_org_labels = pd.DataFrame(clus_org.labels_)

        clustering = []
        clus_labs = clus_org_labels
        clus_labs = clus_labs.rename(columns={0:"orglabs"})
        sil_coeffs = []
        
        for i in range(len(self.model.layers)):
            print("Clustering of layer " + str(i))
            clus_temp = clus.fit(self.dfs[i])
            clustering.append(clus_temp)
        
            #Gaussian Mixture returns the labels differently
            if self.method == "gauss":
                labs_temp = pd.DataFrame(clus.predict(self.dfs[i]))
                labs_temp = labs_temp.rename(columns={0: i})
            else:
                labs_temp = pd.DataFrame(clus_temp.labels_)
                labs_temp = labs_temp.rename(columns={0: i})
            clus_labs = pd.concat([clus_labs, labs_temp], axis=1)
            
            
            #If only one cluster, the silhouette score can't be predicted
            if len(clus_labs.iloc[:,i+1].unique())==1:
                sil_score= "NA"
            else:
                sil_score = silhouette_score(self.dfs[i], np.ravel(labs_temp), metric='euclidean')
            sil_coeffs.append(sil_score)
           
        #Compture NMI
        nmi = np.zeros((clus_labs.shape[1],clus_labs.shape[1]))
        for i in range(clus_labs.shape[1]):
            for j in range(clus_labs.shape[1]):
                nmi[i,j] = normalized_mutual_info_score(clus_labs.iloc[:,i],clus_labs.iloc[:,j])
       
        #Compute ARI
        ari = np.zeros((clus_labs.shape[1],clus_labs.shape[1]))
        for i in range(clus_labs.shape[1]):
            for j in range(clus_labs.shape[1]):
                ari[i,j] = adjusted_rand_score(clus_labs.iloc[:,i],clus_labs.iloc[:,j])
              
        return clus_labs, sil_coeffs, nmi, ari
    
    
class layer_plots():
    def __init__(self, model, dfs, y_test_org, labs=None, silent=False):
        self.dfs = dfs
        self.model = model
        self.labs = labs
        self.silent = silent
        
        self.y_test_org = pd.DataFrame(y_test_org)
        self.y_test_org = self.y_test_org.rename(columns={self.y_test_org.columns[0]: "class_labs"})
        self.y_test_org = self.y_test_org.astype("int")
       
        
        #Define list of ten colors for the instances and ten for the means
        even = list(range(0, 20, 2))
        odd = list(range(1, 20, 2))
        colors = cm.get_cmap('tab20',20)
        self.colors1=colors(odd)
        self.colors2=colors(even)
        
    #-------------------------------- PC Plots -------------------------------
    def pcplot(self):
        if not self.silent:
            print("\n------------------- PC plot of each layer -------------------")

        df_temp = []
        n_subfigs = len(self.model.layers)-1
        fig = plt.figure(constrained_layout=True, figsize=(16,18), dpi=120)
        subfigs = fig.subfigures(n_subfigs)
        fig.suptitle("Parallel Coordinate Plot of Hidden Layers", fontsize='xx-large')
        for i in range(len(self.model.layers)-1):
            if not self.silent:
                print("\n------------ Layer " + str(i) + " ------------")
            dff = pd.DataFrame(self.dfs[i])
            df_temp.append(pd.concat([dff, self.y_test_org.set_index(dff.index)], axis=1)) 
            
            unitmeans = pd.DataFrame()
            
            #get unique true label values
            uniqVal = self.y_test_org["class_labs"].unique()
            if not self.silent:
                print(str(len(uniqVal)) + " classes present")

            for k in uniqVal:
                unitmean = df_temp[i][df_temp[i]["class_labs"]==k].mean(axis=0)
                if not unitmeans.empty:
                    if not self.silent:
                        print("\t next mean computed")
                    unitmean = pd.DataFrame(unitmean)
                    unitmeans = pd.concat([unitmeans,pd.DataFrame(unitmean).T], ignore_index=True)
                else:
                    if not self.silent:
                        print("\t First mean computed")
                    unitmeans = pd.DataFrame(unitmean)
                    unitmeans = unitmeans.T
                unitmeans["class_labs"] = unitmeans["class_labs"].replace(k,"mean "+str(int(k)))  
            unitmeans = pd.DataFrame(unitmeans)   
            
            #construct parallel coordinate plot for current layer
            ax = subfigs[i].subplots(1)
            ax.set_title("Layer " + str(i+1) + " of " + str(len(self.model.layers)))
            parallel_coordinates(pd.DataFrame(df_temp[i]), "class_labs", color=self.colors1, ax=ax)
          
            #If only one class is present in the cluster, else...
            if unitmeans.shape[1] == 1:
                print("OBS: only one class is present in one of the clusters in layer "+str(i))
            else:
                parallel_coordinates(unitmeans, "class_labs", color=self.colors2, ax=ax)
                
            ax.legend(bbox_to_anchor=(0.5, -0.32), ncol=(len(uniqVal)*2), loc='lower center')
        
        return fig
            
    #----------------------------- PC Plots kmeans----------------------------
    
    def pcplot_cluster_layers(self):
        
        print("\n----------- PC plot of each cluster in each layer -----------")
        
        figs = [] 

        #Finding min and max value of the neurons for each layer
        minVal = []
        maxVal = []
        for i in range(len(self.model.layers)-1):
            minTemp = []
            maxTemp = []
            for n in range(len(self.dfs[i][0])):
                minTemp.append(min(self.dfs[i][:,n]))
                maxTemp.append(max(self.dfs[i][:,n]))
            minVal.append(min(minTemp))
            maxVal.append(max(maxTemp))

        
        for i in range(len(self.model.layers)-1):
            if not self.silent:
                print("\n------------ Layer " + str(i) + " ------------")
                
            #Get number of clusters
            num_clus = max(self.labs.iloc[:,i])+1    
                
            #n_subfigs = len(self.model.layers)-1
            fig = plt.figure(constrained_layout=True, figsize=(10,20), dpi=120)
            subfigs = fig.subfigures(num_clus,1, wspace = 0.25)
            fig.suptitle("Parallel Coordinate Plot of Each Cluster in Hidden Layer "+str(i), fontsize='xx-large')
            
            #Get labels for the current layer and concat with true class labels
            y_lay = pd.DataFrame(self.labs.iloc[:,i])
            if i == 0:
                y_lay = y_lay.rename(columns={"orglabs":"labs"})
            else:
                y_lay = y_lay.rename(columns={i-1:"labs"})
            self.y_test_org.reset_index(drop=True, inplace=True)
            data_lay = pd.concat([pd.DataFrame(self.dfs[i]),y_lay, self.y_test_org], axis=1)

            #Loop through all clusters
            for j in range(max(self.labs.iloc[:,i])+1):
                clus = data_lay[data_lay["labs"]==j]  
                clus = clus.drop(["labs"], axis=1)
                
                #get unique true label values in current cluster
                uniqVal = clus["class_labs"].unique()
                if not self.silent:
                    print(str(len(uniqVal)) + " classes present in cluster number " + str(j+1))
            
                #loop through unique unique true label values in current cluster and
                # compute mean of each for each (neuron?)
                colmeans = pd.DataFrame()
                for k in uniqVal:
                    colmean = clus[clus["class_labs"]==k].mean(axis=0)
                    if not colmeans.empty:
                        if not self.silent:
                            print("\t next mean computed")
                        colmean = pd.DataFrame(colmean)
                        colmean = colmean.rename(columns={0: "clus " + str(k)})
                        colmeans = pd.concat([colmeans,pd.DataFrame(colmean).T], ignore_index=True)
                    else:
                        if not self.silent:
                            print("\t First mean computed")
                        colmeans = pd.DataFrame(colmean)
                        colmeans = colmeans.T
                        #colmeans = colmeans.rename(columns={0: "clus " + str(k)})
                    colmeans["class_labs"] = colmeans["class_labs"].replace(k,"mean "+str(int(k)))  
                colmeans = pd.DataFrame(colmeans)    
                
                ax = subfigs[j].subplots(1)

                #construct parallel coordinate plot for current layer and cluster
                cols1 = [self.colors1[uniqVal[k]] for k in range(len(uniqVal))]
                parallel_coordinates(clus, "class_labs", color=cols1, ax=ax)
                
                #colors for means
                cols2 = [self.colors2[uniqVal[k]] for k in range(len(uniqVal))]
                
                #If only one class is present in the cluster, else...
                if colmeans.shape[1] != 1:
                    parallel_coordinates(colmeans, "class_labs", color=cols2, ax=ax)
                ax.set_ylim([minVal[i],maxVal[i]]) 
                ax.legend(title="Cluster " + str(j+1)+"\nClasses:",bbox_to_anchor=(1,1), loc="upper left")
                figs.append(fig)

        return figs
    

    def pcplot_cluster_means(self):
        if not self.silent:
            print("\n------------------- PC plot of each layer -------------------")
        n_subfigs = len(self.model.layers)-1
        fig = plt.figure(constrained_layout=True, figsize=(16,18), dpi=120)
        subfigs = fig.subfigures(n_subfigs)
        fig.suptitle("Parallel Coordinate Plot of Hidden Layers", fontsize='xx-large')
        
        
        for i in range(len(self.model.layers)-1):
            
            #Get labels for the current layer and concat with true class labels
            data_lay = pd.concat([pd.DataFrame(self.dfs[i]),self.labs], axis=1)
            
            if not self.silent:
                print("\n------------ Layer " + str(i) + " ------------")
            
            #Get labels for the current layer and concat with true class labels
            y_lay = pd.DataFrame(self.labs.iloc[:,i])
            if i == 0:
                y_lay = y_lay.rename(columns={"orglabs":"labs"})
            else:
                y_lay = y_lay.rename(columns={i-1:"labs"})
            self.y_test_org.reset_index(drop=True, inplace=True)
            data_lay = pd.concat([pd.DataFrame(self.dfs[i]),y_lay, self.y_test_org], axis=1)
            
            #set plot
            subfigs[i].suptitle("Layer " + str(i+1) + " of " + str(len(self.model.layers)), fontsize='x-large')
            
            #Loop through all clusters
            for j in range(max(self.labs.iloc[:,i])+1):
                
                #get unique true label values in current cluster
                uniqVal = data_lay["labs"].unique()
                if not self.silent:
                    print(str(len(uniqVal)) + " classes present in cluster number " + str(j+1))
            
                #loop through unique true label values in current cluster and
                # compute mean of each for each cluster
                colmeans = pd.DataFrame()
                for k in uniqVal:
                    colmean = data_lay[data_lay["labs"]==k].mean(axis=0)
                    if not colmeans.empty:
                        if not self.silent:
                            print("\t next mean computed")
                        colmean = pd.DataFrame(colmean)
                        colmean = colmean.rename(columns={0: "clus" + str(k)})
                        colmeans = pd.concat([colmeans,pd.DataFrame(colmean).T], ignore_index=True)
                    else:
                        if not self.silent:
                            print("\t First mean computed")
                        colmeans = pd.DataFrame(colmean)
                        colmeans = colmeans.T
                    colmeans["labs"] = colmeans["labs"].replace(k,"mean "+str(int(k)))  
                colmeans = pd.DataFrame(colmeans) 
            
            colmeans = colmeans.drop(["class_labs"], axis=1)
            
            #construct parallel coordinate plot for current layer
            ax = subfigs[i].subplots(1)

            #If only one class is present in the cluster, else...
            if colmeans.shape[1] == 1:
                print("OBS: only one class is present in one of the clusters in layer "+str(i))
            else:
                parallel_coordinates(colmeans, "labs", color=self.colors2, ax=ax)
                
            ax.legend(bbox_to_anchor=(0.5, -0.32), ncol=(len(uniqVal)*2), loc='lower center')
            
        return fig
            


class epoch_plots():
    def __init__(self, model, epoch_outputs, y_test_org, labs=None, silent=False):
        self.epoch_outputs = epoch_outputs
        self.model = model
        self.labs = labs
        self.silent = silent
        
        self.y_test_org = pd.DataFrame(y_test_org)
        self.y_test_org = self.y_test_org.rename(columns={self.y_test_org.columns[0]: "class_labs"})
        self.y_test_org = self.y_test_org.astype("int")
        
        #Define list of ten colors for the instances and ten for the means
        even = list(range(0, 20, 2))
        odd = list(range(1, 20, 2))
        colors = cm.get_cmap('tab20',20)
        self.colors1=colors(odd)
        self.colors2=colors(even)
        
    #-------------------------------- PC Plots -------------------------------
    def pcplot(self):
        
        if not self.silent:
            print("\n------------------- PC plot of each layer -------------------")
        
        n_epochs = len(self.epoch_outputs)
        
        #Finding min and max value of the neurons for each layer across the epochs
        minVal = []
        maxVal = []
        for j in range(n_epochs):
            # print(j)
            minTempVal = []
            maxTempVal = []
            for i in range(len(self.epoch_outputs[j])):
                minTemp = []
                maxTemp = []
                for n in range(len(self.epoch_outputs[j][i][0])):
                    # print(j,i,n)
                    minTemp.append(min(self.epoch_outputs[j][i][:,n]))
                    maxTemp.append(max(self.epoch_outputs[j][i][:,n]))
                minTempVal.append(min(minTemp))
                maxTempVal.append(max(maxTemp))
            minVal.append(minTempVal)
            maxVal.append(maxTempVal)
        
        pdMin = pd.DataFrame(minVal)
        pdMax = pd.DataFrame(maxVal)
        minn = []
        maxx = []
        for i in range(len(pdMax.iloc[0])):
            minn.append(min(pdMin.iloc[:,i]))
            maxx.append(max(pdMax.iloc[:,i])) 

        figs = []
        for j in range(n_epochs):
            if not self.silent:
                print("\n---------------------------------" )
                print("------------ Epoch " + str(j) + " -----------")
                print("---------------------------------" )
            df_temp = []
            epoch_dfs = self.epoch_outputs[j]
            n_subfigs = len(self.model.layers)-1
            fig = plt.figure(constrained_layout=True, figsize=(10,10), dpi=120)
            subfigs = fig.subfigures(n_subfigs)
            fig.suptitle("Parallel Coordinate Plot of Hidden Layers in Epoch "+str(j+1)+" of "+str(n_epochs), fontsize='xx-large')
            for i in range(len(self.model.layers)-1):
                if not self.silent:
                    print("\n------------ Layer " + str(i) + " ------------")
                dff = pd.DataFrame(epoch_dfs[i])
                df_temp.append(pd.concat([dff, self.y_test_org.set_index(dff.index)], axis=1)) 
                
                unitmeans = pd.DataFrame()
                
                #get unique true label values
                uniqVal = self.y_test_org["class_labs"].unique()
                if not self.silent:
                    print(str(len(uniqVal)) + " classes present")
                
                for k in uniqVal:
                    unitmean = df_temp[i][df_temp[i]["class_labs"]==k].mean(axis=0)
                    if not unitmeans.empty:
                        if not self.silent:
                            print("\t next mean computed")
                        unitmean = pd.DataFrame(unitmean)
                        unitmeans = pd.concat([unitmeans,pd.DataFrame(unitmean).T], ignore_index=True)
                    else:
                        if not self.silent:
                            print("\t First mean computed")
                        unitmeans = pd.DataFrame(unitmean)
                        unitmeans = unitmeans.T
                    unitmeans["class_labs"] = unitmeans["class_labs"].replace(k,"mean "+str(int(k)))  
                unitmeans = pd.DataFrame(unitmeans)   
                
                #construct parallel coordinate plot for current layer
                ax = subfigs[i].subplots(1)
                ax.set_title("Layer " + str(i+1) + " of " + str(len(self.model.layers)))
                ax.set_xlabel("Neurons")
                parallel_coordinates(pd.DataFrame(df_temp[i]), "class_labs", color=self.colors1, ax=ax)
              
                #If only one class is present in the cluster, else...
                if unitmeans.shape[1] == 1:
                    print("OBS: only one class is present in one of the clusters in layer "+str(i))
                    
                else:
                    parallel_coordinates(unitmeans, "class_labs", color=self.colors2, ax=ax)
                
                ax.set_ylim([minn[i],maxx[i]]) 
                ax.legend(bbox_to_anchor=(1.01,1.1), ncol=1, loc='upper left')
            figs.append(fig)

        return figs
  

class cluster_measures():
    def __init__(self, layer_meas, epochs, silent=False):
        self.layer_meas = layer_meas
        self.epochs = epochs
        self.silent = silent
        
    def mean_layer_meas(self):
        mean_vec = []
        for i in range(len(self.layer_meas[0])):
            temp = 0
            for j in range(len(self.layer_meas)):
                temp = temp+self.layer_meas[j][i]
            mean_vec.append(temp/len(self.layer_meas))
        return mean_vec
    
    def mean_layer_meas_no_out(self):
        mean_vec = []
        for i in range(len(self.layer_meas[0])):
            temp = 0
            for j in range(len(self.layer_meas)-1):
                temp = temp+self.layer_meas[j][i]
            mean_vec.append(temp/(len(self.layer_meas)-1))
        return mean_vec
    
    def moving_block_mean(self, mean_vec):
        #Step 1
        bsize = np.floor(self.epochs/10)
        beg = 0
        end = int(beg+bsize)
        bmean = []
        temp_mean_best = 0
        best_beg = 0
        for i in range(int(self.epochs-bsize)):
            temp_mean = sum(mean_vec[beg:end])/bsize
            bmean.append(temp_mean)
            if temp_mean>temp_mean_best:
              best_beg = i
              temp_mean_best = temp_mean
            beg = beg+1
            end = end+1
            
        #Step2
        bsize2 = np.floor(bsize/2)
        bbeg = best_beg
        bend = int(best_beg+bsize2)
        bmean2 = []
        temp_mean_best2 = 0
        best_beg2 = 0
        
        for i in range(int(bsize-bsize2)):
            temp_mean = sum(mean_vec[bbeg:bend])/bsize2
            bmean2.append(temp_mean)
            if temp_mean>temp_mean_best2:
                best_beg2 = best_beg+i
                temp_mean_best2 = temp_mean
            bbeg = bbeg + 1
            bend = bend + 1
        
        #best_end2 = int(best_beg2 + bsize2)
        best_epoch = int(np.floor(best_beg2+bsize2/2))
        
        return best_epoch
        
        
    def moving_block_meansd(self, mean_vec):
        #Step 1a - mean
        bsize = np.floor(self.epochs/5)
        beg = 0
        end = int(beg+bsize)
        temp_mean_best = [0]*int(np.floor((bsize/2)))
        best_beg = [0]*int(np.floor((bsize/2)))
        for i in range(int(self.epochs-bsize)):
            temp_mean = sum(mean_vec[beg:end])/bsize
            for j in range(len(temp_mean_best)):
                if temp_mean>temp_mean_best[j]:
                  best_beg[j] = i
                  temp_mean_best[j] = temp_mean
                  break
            beg = beg+1
            end = end+1
        
        #Step 1b - sd
        best_beg_sd = 0
        temp_sd_best = 1000
        for i in range(len(best_beg)):
            temp_end = int(best_beg[i]+bsize)
            temp_sd = np.std(mean_vec[best_beg[i]:temp_end])
            if temp_sd<temp_sd_best:
                temp_sd_best = temp_sd
                best_beg_sd = best_beg[i]
         
        #Step 2a - mean
        bsize2 = np.floor(bsize/2)
        beg = best_beg_sd
        end = int(beg+bsize2)
        temp_mean_best2 = [0]*int(np.floor((bsize2/2)))
        best_beg2 = [0]*int(np.floor((bsize2/2)))
        for i in range(int(bsize-bsize2+1)):
            temp_mean = sum(mean_vec[beg:end])/bsize2
            for j in range(len(temp_mean_best2)):
                if temp_mean>temp_mean_best2[j]:
                    best_beg2[j] = best_beg_sd+i
                    temp_mean_best2[j] = temp_mean
                    break
            beg = beg+1
            end = end+1
            
        #Step 2b - sd
        best_beg_sd2 = 0
        temp_sd_best2 = 1000
        for i in range(len(best_beg2)):
            temp_end = int(best_beg2[i]+bsize2)
            temp_sd = np.std(mean_vec[best_beg2[i]:temp_end])
            if temp_sd<temp_sd_best2:
                temp_sd_best2 = temp_sd
                best_beg_sd2 = best_beg2[i]
        
        best_epoch = int(np.floor(best_beg_sd2+bsize2/2))
        
        return best_epoch


    def moving_block_sdmean(self, mean_vec):
        #Step 1a - sd   
        bsize=np.floor(self.epochs/5)
        beg = 0
        end = int(beg+bsize)
        temp_sd_best = [1000]*int(np.floor((bsize/2)))
        best_beg = [0]*int(np.floor((bsize/2)))
        for i in range(int(self.epochs-bsize)):
            temp_sd = np.std(mean_vec[beg:end])
            for j in range(len(temp_sd_best)):
                if temp_sd < temp_sd_best[j]:
                    best_beg[j] = i
                    temp_sd_best[j] = temp_sd
                    break
                beg = beg +1 
                end = end +1
        
        #Step 1b - mean
        best_beg_mean = 0
        temp_mean_best = 0
        for i in range(len(best_beg)):
            temp_end = int(best_beg[i]+bsize)
            temp_mean = sum(mean_vec[best_beg[i]:temp_end])/bsize
            if temp_mean>temp_mean_best:
                temp_mean_best = temp_mean
                best_beg_mean = best_beg[i]
       
        
        #Step 2a - sd 
        bsize2 = np.floor(bsize/2)
        beg = best_beg_mean
        end = int(beg+bsize2)
        temp_sd_best2 = [1000]*int(np.floor((bsize2/2)))
        best_beg2 = [0]*int(np.floor((bsize2/2)))
        for i in range(int(bsize-bsize2+1)):
            temp_sd = np.std(mean_vec[beg:end])
            for j in range(len(temp_sd_best2)):
                if temp_sd<temp_sd_best2[j]:
                    best_beg2[j] = best_beg_mean+i
                    temp_sd_best2[j] = temp_sd
                    break
            beg = beg+1
            end = end+1
            
        #Step 2b - mean
        best_beg_mean2 = 0
        temp_mean_best2 = 0
        for i in range(len(best_beg2)):
            temp_end = int(best_beg2[i]+bsize2)
            temp_mean = sum(mean_vec[best_beg2[i]:temp_end])/bsize
            if temp_mean>temp_mean_best2:
                temp_mean_best2 = temp_mean
                best_beg_mean2 = best_beg2[i]
        
        best_epoch = int(np.floor(best_beg_mean2+bsize2/2))
        
        return best_epoch
