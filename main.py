# -*- coding: utf-8 -*-
"""
Created on Aug 13 2022
Last edited on Mar 22 2023
@author: carol

This file is developed for the master's thesis "Clustering Based Explanations
of Decisions Structures in Deep Neural Networks". It contains the main code
for the data management, the network construction, the extraction of weigts
from the networks and the analyses of those. The code is constructed in a way
such that it is automated to run for all datasets and all clustring methods 
in one run. 
 
This file uses functions from classes defined in the file 'classes.py'.

"""

#Set wd
import os
path = "C:/Users/carol/OneDrive/10semester/Speciale/handin"
os.chdir(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle

#Scaling and encoding of labels
from sklearn.preprocessing import LabelEncoder, StandardScaler
labEnc = LabelEncoder()
stdScal = StandardScaler()

#Test/train-split
from sklearn.model_selection import train_test_split

#Network
from tensorflow import keras
from tensorflow.keras import layers

#Movie from pictures
import moviepy.video.io.ImageSequenceClip

#Costum classes
from classes import layer_outputs, clustering_layers, layer_plots, CustomCallback, epoch_plots, cluster_measures

#File handeling
import pickle

#Clusterings
import scipy.cluster.hierarchy as hier_clus
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

#Plotting
import seaborn as sn

#Links
#(hierarchical) agglomerative clustering:
#https://www.section.io/engineering-education/hierarchical-clustering-in-python/

#EM-clustering/Gaussian Mixture model
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
#from sklearn.mixture import GaussianMixture

#-----------------------------------------------------------------------------
#------------------------------- Initialization ------------------------------
#-----------------------------------------------------------------------------

choose_num_clus = False

#Cluster methods to loop through
clus_list = [ "kmeans", "hdbscan", "gauss", "agglomerative"]
#clus_list = ["kmeans"]
         
#Choice of data(#classes): mhr(3), haber(2), mamo(2), cmc(3), obs(7)
data_list = ["mhr", "haber", "mamo", "cmc"]
#data_list = ["mhr"]

for dataset in data_list: 

    #---------------------------------- mhr Data ---------------------------------
    if dataset == "mhr":
        data = pd.read_csv('data/mhr.csv')

    #------------------------------ Mamographic Data -----------------------------
    if dataset == "mamo":
        data_org = pd.read_csv('data/mammographic_masses.data', header = None, na_values='?')
        data = data_org.dropna(axis=0, how='any')
   
    #---------------------------------- obs Data ---------------------------------
    if dataset == "obs":
        data_org = pd.read_csv('data/obesity.csv')
        data = data_org.dropna(axis=0, how='any')
        
        for column in data.columns:
            if data.dtypes[column]=='O':
                data[column] = labEnc.fit_transform(data[column].astype('str'))
        
    #--------------------------------- Data prep ---------------------------------
    #Prepare data
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1:]
    y = y.rename(columns={y.columns[0]: "class_labs"})
    
    
    #Define number of classes in data
    num_classes = len(y["class_labs"].unique())
    y_str = y
              
    #np.ravel used to make y_str on right dimensions
    y_enc = labEnc.fit_transform(np.ravel(y_str))
    y = keras.utils.to_categorical(y_enc)
    
    # Test-train split
    X_train_temp, X_test_org, y_train_temp, y_test = train_test_split(x, y, test_size=0.3, stratify=(y_enc), random_state=1)
    X_train_org, X_val_org, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.33, stratify=(y_train_temp), random_state=1)
    
    if dataset != "obs":
        #scaling after split because test data should be new unseen data - bias if used for scaling
        X_train = stdScal.fit_transform(X_train_org)
        X_val = stdScal.fit_transform(X_val_org)
        X_test = stdScal.transform(X_test_org)
    
        x_scal = stdScal.fit_transform(x)
    else:
        X_train = X_train_org
        X_val = X_val_org
        X_test = X_test_org
        
        
    # make list consisting of true labels of val set
    orglabs_val = []
    for i in range(len(y_val[0])):
        if i==0:
            orglabs_val = y_val[:,0]
        else:
            orglabs_val = orglabs_val + y_val[:,i]*(i+1)
    orglabs_val = orglabs_val-1
    y_val_org = pd.DataFrame(orglabs_val)
    y_val_org = y_val_org.rename(columns={y_val_org.columns[0]: "diag"})
    y_val_org = y_val_org.astype("int")
    
    # make list consisting of true labels of test set
    orglabs = []
    for i in range(len(y_test[0])):
        if i==0:
            orglabs = y_test[:,0]
        else:
            orglabs = orglabs + y_test[:,i]*(i+1)
    orglabs = orglabs-1
    y_test_org = pd.DataFrame(orglabs)
    y_test_org = y_test_org.rename(columns={y_test_org.columns[0]: "diag"})
    y_test_org = y_test_org.astype("int")
    
 
    #Create folder for files
    data_dir = "results/"+dataset
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    #If we want to determine number of clusters before running entire code
    if choose_num_clus:
        #clus_list = ["kmeans", "agglomerative"]
        for clus_method in clus_list:
            #Create folder for files
            clus_dir = data_dir+"/"+clus_method
            if not os.path.isdir(clus_dir):
                os.makedirs(clus_dir)
                
            #finding the optimal number of clusters using elbow plot
            if clus_method == "kmeans":
                #we have to select the value of k at the “elbow” ie the point 
                #after which the distortion/inertia start decreasing in a linear 
                #fashion
                K = range(1,20)
                inerti = []
                for k in K:
                    km_model = KMeans(n_clusters=k)
                    km_model.fit(X_test)
                    inerti.append(km_model.inertia_)
                plt.figure(figsize=(8,6), dpi=120)    
                plt.plot(K, inerti)
                plt.xticks(K)
                plt.xlabel('k')
                plt.ylabel('inerti')
                plt.title('Elbow Plot for K-means\n Dataset: '+dataset)
                plt.savefig(clus_dir+"/elbow.png")
                print("OBS: Choose num_clus from elbow plot!")
                
            #finding the optimal number of clusters using dendrogram
            if clus_method == "agglomerative":
                plt.figure(figsize=(8,6), dpi=120)
                dendrogram = hier_clus.dendrogram(hier_clus.linkage(X_test, method = 'ward'),no_labels=True) 
                plt.title('Dendrogram\nDataset: '+dataset)
                plt.xlabel('Data instances') 
                plt.ylabel('Euclidean distances') 
                plt.savefig(clus_dir+"/dendrogram.png")
                print("OBS: Choose num_clus from dendrogram!")
     
    
    #-----------------------------------------------------------------------------
    #---------------------------------- NN models --------------------------------
    #-----------------------------------------------------------------------------
    
    #If we want to run the code with predefined number of clusters
    elif not choose_num_clus:
        # define the keras models
        if dataset == "mhr":
            model = keras.Sequential()
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(num_classes, activation='softmax'))
            
            # epochs = 30
            epochs=100
            batch_size=32
            
        if dataset == "mamo":
            model = keras.Sequential()
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            # model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(num_classes, activation='softmax'))
            
            #epochs = 10
            epochs=100
            batch_size=32
            
        if dataset == "obs":
            model = keras.Sequential()
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(num_classes, activation='sigmoid'))
            
            #epochs = 10
            epochs=500
            batch_size=32

    
        #-----------------------------------------------------------------------------
        
        #Compile the keras model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
        #Output of each layer
        for i, layer in enumerate(model.layers):
            layer._name = 'layer_' + str(i)
        
        #Use custom callback to get data after each epoch
        cusCallback = CustomCallback(X_test, y_test, model, data_dir)
        
        #Fit the keras model on the dataset
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_val, y_val), callbacks=[cusCallback])

        #Load output of each layer in each epoch from pikle objects
        epoch_outputs = []
        for i in range(epochs):
            with open(data_dir+'/df_'+str(i)+'.pkl', 'rb') as in_file:
                temp = pickle.load(in_file)
                epoch_outputs.append(temp)
            os.remove(data_dir+'/df_'+str(i)+'.pkl')
            
        #Load output of each layer in each epoch from pikle objects
        test_eval = []
        for i in range(epochs):
            with open(data_dir+'/test_eval_'+str(i)+'.pkl', 'rb') as in_file:
                temp2 = pickle.load(in_file)
                test_eval.append(temp2)
            os.remove(data_dir+'/test_eval_'+str(i)+'.pkl')
            
        test_pd = pd.DataFrame(test_eval)     
        test_pd = test_pd.rename(columns={0: "loss", 1: "accuracy"})
        test_pd.to_excel(data_dir+"/test_eval.xlsx", header=False, index=False)        

        #Save training history
        with pd.ExcelWriter(data_dir+"/model_hist.xlsx") as writer:  
            pd.DataFrame(history.history['accuracy']).to_excel(writer, header=False, index=False, sheet_name="train_acc")
            pd.DataFrame(history.history['val_accuracy']).to_excel(writer, header=False, index=False, sheet_name="val_acc")
            pd.DataFrame(history.history['loss']).to_excel(writer, header=False, index=False, sheet_name="train_loss")
            pd.DataFrame(history.history['val_loss']).to_excel(writer, header=False, index=False, sheet_name="val_loss")
            
        # summarize history for accuracy
        plt.figure(figsize=(8,6), dpi=120)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(test_pd["accuracy"])
        plt.title("Model Accuracy\n Dataset: "+dataset)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_accuracy.png")
        plt.show()
        
        # summarize history for loss
        plt.figure(figsize=(8,6), dpi=120)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(test_pd["loss"])
        plt.title("Model Loss\n Dataset: "+dataset)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_loss.png")
        plt.show()
        
        #-----------------------------------------------------------------------------
        #------------------------------- Layer Outputs -------------------------------
        #-----------------------------------------------------------------------------
        
        #Save outputs of each layer
        outputs = layer_outputs(model, X_test)
        dfs, intermediate_outputs = outputs.getIntermediateOutputs()
        
        #-----------------------------------------------------------------------------
        #----------------------------- PC Plots of Layers ----------------------------
        #-----------------------------------------------------------------------------
        
        #PCPlot of each layers after training with cluster labels
        if num_classes==2:
            layer_plot = layer_plots(model, dfs, y_test)
        else:
            layer_plot = layer_plots(model, dfs, y_test_org)
        fig = layer_plot.pcplot()
        fig.savefig(data_dir+"/after_train_"+dataset+".png")
        
        plt.close('all')
        
        #-----------------------------------------------------------------------------
        #--------------------------- Clustering of Layers ----------------------------
        #-----------------------------------------------------------------------------
        
        cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        epoch_dic_ari = {}
        epoch_dic_nmi = {}
        
        for clus_method in clus_list:
            #Create folder for files
            clus_dir = data_dir+"/"+clus_method
            if not os.path.isdir(clus_dir):
                os.makedirs(clus_dir)
             
            #Set predefined number of clusters for k-means
            if clus_method == "kmeans":
                if num_classes>2:
                    ks = [int(np.ceil(num_classes/2)),num_classes, num_classes*2]
                else: 
                    ks = [num_classes, num_classes*2]

                for num_clus in ks:
                    #Create folder for files
                    km_dir = clus_dir+"/"+str(num_clus)
                    if not os.path.isdir(km_dir):
                        os.makedirs(km_dir)
                    
                    #clustering on each layer
                    if num_classes==2:
                        clusteringLayers = clustering_layers(clus_method, model, dfs, num_clus, X_test, y_test)
                    else:
                        clusteringLayers = clustering_layers(clus_method, model, dfs, num_clus, X_test, y_test_org)

                    labs, sil_coeffs, nmi, ari = clusteringLayers.get_clusters()
                    
                    #Save informations from clustering
                    pd.DataFrame(sil_coeffs).to_csv(km_dir+"/sil_coeffs.csv", sep=';', header=False, index=False)
                    pd.DataFrame(nmi).to_csv(km_dir+"/nmi.csv", sep=';', header=False, index=False)
                    pd.DataFrame(ari).to_csv(km_dir+"/ari.csv", sep=';', header=False, index=False)

                    #-----------------------------------------------------------------------------
                    #---------------------- PC Plots of Clustering of Layers ---------------------
                    #-----------------------------------------------------------------------------
                    if dataset != "obs":
                        #PCPlot of each layers after training with cluster labels
                        if num_classes==2:
                            layer_plot = layer_plots(model, dfs, y_test, labs)
                        else:
                            layer_plot = layer_plots(model, dfs, y_test_org, labs)
                            
                        #PCPlot of each cluster after training with true class labels
                        fig2 = layer_plot.pcplot_cluster_layers()
                        for i in range(len(fig2)):
                            fig2[i].savefig(km_dir+"/cluster_plots_lay_"+str(i)+".png")
                        del fig2
                        
                        #PCPlot of cluster means in each layers
                        fig3 = layer_plot.pcplot_cluster_means()
                        fig3.savefig(km_dir+"/clusterMeans.png")
                        del fig3
                        
                        plt.close('all')
                    
                    # #-----------------------------------------------------------------------------
                    # #----------------------------- PC Plots of Epochs ----------------------------
                    # #-----------------------------------------------------------------------------
                    if dataset != "obs":
                        #PCPlot of each layers after each epoch during training with true class labels
                        if num_classes==2:
                            epoch_plot = epoch_plots(model, epoch_outputs, y_test, silent = False)
                        else: 
                            epoch_plot = epoch_plots(model, epoch_outputs, y_test_org, silent = False)
                        figs = epoch_plot.pcplot()
            
                    
                    #Create folder if it does not already exist
                    epoch_dir = km_dir+"/epoch_plots"
                    if not os.path.isdir(epoch_dir):
                        os.makedirs(epoch_dir)
                        
                    if dataset != "obs":
                        #Save plot of layers in each epoch - OBS no clustering
                        for i in range(len(figs)):
                            figs[i].savefig(epoch_dir+"/epoch_"+str(i)+".png")
                        
                        #Epoch plots to movie
                        fps=3 #files pr. second
                        image_files = [os.path.join(epoch_dir,"epoch_"+str(k)+".png") 
                                        for k in range(len(figs))]
                        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                        clip.write_videofile(epoch_dir+'/epoch_movie.mp4')
                        
                        for i in range(len(figs)):
                            os.remove(epoch_dir+"/epoch_"+str(i)+".png")
                        
                        plt.close('all')
                        
                    #-----------------------------------------------------------------------------
                    #--------------------------- Clustering of Epochs ----------------------------
                    #-----------------------------------------------------------------------------
                    
                    epoch_labs = []
                    epoch_sil_coeffs = []
                    epoch_nmi = []
                    epoch_ari = [] 
                    
                    #Go through data from each epoch   
                    for i in range(len(epoch_outputs)):
                        #clustering on each layer 
                        if num_classes==2:
                            clusteringLayers = clustering_layers(clus_method, model, epoch_outputs[i], num_clus, X_test, y_test)
                        else:
                            clusteringLayers = clustering_layers(clus_method, model, epoch_outputs[i], num_clus, X_test, y_test_org)
                            
                        temp_labs, temp_sil_coeffs, temp_nmi, temp_ari = clusteringLayers.get_clusters()
                        
                        epoch_labs.append(temp_labs)
                        epoch_sil_coeffs.append(temp_sil_coeffs)
                        epoch_nmi.append(temp_nmi)
                        epoch_ari.append(temp_ari)
                        
                    #Save informations from clustering
                    with pd.ExcelWriter(epoch_dir+"/sil_coeff_epoch.xlsx") as writer:  
                        for i in range(len(epoch_outputs)):
                            pd.DataFrame(epoch_sil_coeffs[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))
                    with pd.ExcelWriter(epoch_dir+"/nmi_epoch.xlsx") as writer:  
                        for i in range(len(epoch_outputs)):
                            pd.DataFrame(epoch_nmi[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))
                    with pd.ExcelWriter(epoch_dir+"/ari_epoch.xlsx") as writer:  
                        for i in range(len(epoch_outputs)):
                            pd.DataFrame(epoch_ari[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))
                    
                    #Sim measures to plot
                    layer_sim_meas_ari = np.zeros((len(epoch_labs[0].iloc[0])-1,len(epoch_labs)-1))
                    layer_sim_meas_nmi = np.zeros((len(epoch_labs[0].iloc[0])-1,len(epoch_labs)-1))
                    for i in range(len(epoch_labs[0].iloc[0])-1):
                        for j in range(len(epoch_labs)-1):
                            layer_sim_meas_ari[i,j] = adjusted_rand_score(epoch_labs[j].iloc[:,i+1],epoch_labs[j+1].iloc[:,i+1])
                            layer_sim_meas_nmi[i,j] = normalized_mutual_info_score(epoch_labs[j].iloc[:,i+1],epoch_labs[j+1].iloc[:,i+1])

                    #Get mean of ech cluster measure across the layers for each epoch
                    clus_meas_ari = cluster_measures(layer_sim_meas_ari, epochs=epochs)
                    mean_ari = clus_meas_ari.mean_layer_meas_no_out()
                    clus_meas_nmi = cluster_measures(layer_sim_meas_nmi, epochs=epochs)
                    mean_nmi = clus_meas_nmi.mean_layer_meas_no_out()
                    
                    #Get best epoch from moving block approach
                    best_epoch_ari = clus_meas_ari.moving_block_sdmean(mean_ari)
                    #best_epoch_ari = clus_meas_ari.moving_block_meansd(mean_ari)
                    epoch_dic_ari[clus_method+str(num_clus)]=str(best_epoch_ari)
                    best_epoch_nmi = clus_meas_nmi.moving_block_sdmean(mean_nmi)
                    #best_epoch_nmi = clus_meas_nmi.moving_block_meansd(mean_nmi)
                    epoch_dic_nmi[clus_method+str(num_clus)]=str(best_epoch_nmi)
                    
                    #Heatmap ARI
                    plt.figure(figsize=(8,6), dpi=120)
                    ax1 = plt.axes()
                    sn.heatmap(layer_sim_meas_ari)
                    ax1.add_patch(Rectangle((best_epoch_ari,0),1,len(model.layers), fill=False, edgecolor='black'))
                    plt.title("ARI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                    plt.ylabel('Layer')
                    plt.xlabel('Epoch')
                    plt.savefig(epoch_dir+"/"+dataset+"_heatmap_ari.png",bbox_inches='tight')
                    
                    #Heatmap NMI
                    plt.figure(figsize=(8,6), dpi=120) 
                    ax1 = plt.axes()
                    sn.heatmap(layer_sim_meas_nmi)
                    ax1.add_patch(Rectangle((best_epoch_nmi,0),1,len(model.layers), fill=False, edgecolor='black'))
                    plt.title("NMI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                    plt.ylabel('Layer')
                    plt.xlabel('Epoch')
                    plt.savefig(epoch_dir+"/"+dataset+"_heatmap_nmi.png",bbox_inches='tight')
                    
                    #Plot ARI
                    legends = []
                    plt.figure(figsize=(8,6), dpi=120)
                    ax1 = plt.axes()
                    ax1.set_ylim([0,1.01])
                    for i in range(len(layer_sim_meas_ari)):
                        legends.append('Layer '+str(i))
                        plt.plot(layer_sim_meas_ari[i], color=cols[i])
                    plt.plot(mean_ari, color="black")
                    legends.append("mean")
                    plt.axvline(x = best_epoch_ari, color = 'black')
                    plt.text(best_epoch_ari, 0.055, str(best_epoch_ari), color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
                    plt.title("ARI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                    plt.ylabel('ARI')
                    plt.xlabel('Epoch')
                    plt.legend(legends, bbox_to_anchor=(1, 1.025), loc='upper left')
                    plt.savefig(epoch_dir+"/"+dataset+"_ari.png",bbox_inches='tight')
                    
                    #Plot NMI
                    legends = []
                    plt.figure(figsize=(8,6), dpi=120)
                    ax1 = plt.axes()
                    ax1.set_ylim([0,1.01])
                    for i in range(len(layer_sim_meas_nmi)):
                        legends.append('Layer '+str(i))
                        plt.plot(layer_sim_meas_nmi[i], color=cols[i])
                    plt.plot(mean_nmi, color="black")
                    legends.append("mean")
                    plt.axvline(x = best_epoch_nmi, color = 'black')
                    plt.text(best_epoch_nmi, 0.055, str(best_epoch_nmi), color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
                    plt.title("NMI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                    plt.ylabel('NMI')
                    plt.xlabel('Epoch')
                    plt.legend(legends, bbox_to_anchor=(1, 1.025), loc='upper left')
                    plt.savefig(epoch_dir+"/"+dataset+"_nmi.png",bbox_inches='tight')

            else:
                #Set predefined number of clusters for Agglomerative clustering
                if clus_method == "agglomerative":
                    if dataset == "mhr":
                        num_clus = 3
                    if dataset == "haber":
                        num_clus = 5 
                    # if dataset == "sepsis":
                    #     num_clus =0
                    if dataset == "mamo":
                        num_clus = 2
                    if dataset == "cmc":
                        num_clus = 2   
                    if dataset == "obs":
                        num_clus = 2
        
                #clustering on each layer
                if num_classes==2:
                    clusteringLayers = clustering_layers(clus_method, model, dfs, num_clus, X_test, y_test)
                else:
                    clusteringLayers = clustering_layers(clus_method, model, dfs, num_clus, X_test, y_test_org)
                labs, sil_coeffs, nmi, ari = clusteringLayers.get_clusters()
                
                #Sim measures of consecutive layers
                ari_cons = []
                nmi_cons = []
                for i in range(len(ari[0])-1):
                    ari_cons.append(ari[i,i+1])
                    nmi_cons.append(nmi[i,i+1])
                
                #Save informations from clustering
                pd.DataFrame(sil_coeffs).to_csv(clus_dir+"/sil_coeffs.csv", sep=';', header=False, index=False)
                pd.DataFrame(nmi).to_csv(clus_dir+"/nmi.csv", sep=';', header=False, index=False)
                pd.DataFrame(ari).to_csv(clus_dir+"/ari.csv", sep=';', header=False, index=False)
                

                #-----------------------------------------------------------------------------
                #---------------------- PC Plots of Clustering of Layers ---------------------
                #-----------------------------------------------------------------------------
                if dataset != "obs":
                    #PCPlot of each layers after training with cluster labels
                    if num_classes==2:
                        layer_plot = layer_plots(model, dfs, y_test, labs)
                    else:
                        layer_plot = layer_plots(model, dfs, y_test_org, labs)
                        
                    #PCPlot of each cluster after training with true class labels
                    fig2 = layer_plot.pcplot_cluster_layers()
                    for i in range(len(fig2)):
                        fig2[i].savefig(clus_dir+"/cluster_plots_lay_"+str(i)+".png")
                    
                    #PCPlot of cluster means in each layers
                    fig3 = layer_plot.pcplot_cluster_means()
                    fig3.savefig(clus_dir+"/clusterMeans.png")
                    
                    plt.close('all')
                
                #-----------------------------------------------------------------------------
                #----------------------------- PC Plots of Epochs ----------------------------
                #-----------------------------------------------------------------------------
                if dataset != "obs":
                    #PCPlot of each layers after each epoch during training with true class labels
                    if num_classes==2:
                        epoch_plot = epoch_plots(model, epoch_outputs, y_test, silent = False)
                    else: 
                        epoch_plot = epoch_plots(model, epoch_outputs, y_test_org, silent = False)
                    figs = epoch_plot.pcplot()
        
                
                #Create folder if it does not already exist
                epoch_dir = clus_dir+"/epoch_plots"
                if not os.path.isdir(epoch_dir):
                    os.makedirs(epoch_dir)
                    
                if dataset != "obs":
                    #Save plot of layers in each epoch - OBS no clustering
                    for i in range(len(figs)):
                        figs[i].savefig(epoch_dir+"/epoch_"+str(i)+".png")
                    
                    #Epoch plots to movie
                    fps=3 #files pr. second
                    image_files = [os.path.join(epoch_dir,"epoch_"+str(k)+".png") 
                                   for k in range(len(figs))]
                    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                    clip.write_videofile(epoch_dir+'/epoch_movie.mp4')
                    
                    
                    for i in range(len(figs)):
                        os.remove(epoch_dir+"/epoch_"+str(i)+".png")
                           
                    plt.close('all')
                
                #-----------------------------------------------------------------------------
                #--------------------------- Clustering of Epochs ----------------------------
                #-----------------------------------------------------------------------------
                
                epoch_labs = []
                epoch_sil_coeffs = []
                epoch_nmi = []
                epoch_ari = [] 
                
                #Go through data from each epoch   
                for i in range(len(epoch_outputs)):
                    #clustering on each layer 
                    if num_classes==2:
                        clusteringLayers = clustering_layers(clus_method, model, epoch_outputs[i], num_clus, X_test, y_test)
                    else:
                        clusteringLayers = clustering_layers(clus_method, model, epoch_outputs[i], num_clus, X_test, y_test_org)
                        
                    temp_labs, temp_sil_coeffs, temp_nmi, temp_ari = clusteringLayers.get_clusters()
                    
                    epoch_labs.append(temp_labs)
                    epoch_sil_coeffs.append(temp_sil_coeffs)
                    epoch_nmi.append(temp_nmi)
                    epoch_ari.append(temp_ari)
                    
                #Save informations from clustering
                with pd.ExcelWriter(clus_dir+"/sil_coeff_epoch.xlsx") as writer:  
                    for i in range(len(epoch_outputs)):
                        pd.DataFrame(epoch_sil_coeffs[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))
                with pd.ExcelWriter(clus_dir+"/nmi_epoch.xlsx") as writer:  
                    for i in range(len(epoch_outputs)):
                        pd.DataFrame(epoch_nmi[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))
                with pd.ExcelWriter(clus_dir+"/ari_epoch.xlsx") as writer:  
                    for i in range(len(epoch_outputs)):
                        pd.DataFrame(epoch_ari[i]).to_excel(writer, header=False, index=False, sheet_name="Epoch_"+str(i))

                
                #Sim measures to plot
                layer_sim_meas_ari = np.zeros((len(epoch_labs[0].iloc[0])-1,len(epoch_labs)-1))
                layer_sim_meas_nmi = np.zeros((len(epoch_labs[0].iloc[0])-1,len(epoch_labs)-1))
                for i in range(len(epoch_labs[0].iloc[0])-1):
                    for j in range(len(epoch_labs)-1):
                        layer_sim_meas_ari[i,j] = adjusted_rand_score(epoch_labs[j].iloc[:,i+1],epoch_labs[j+1].iloc[:,i+1])
                        layer_sim_meas_nmi[i,j] = normalized_mutual_info_score(epoch_labs[j].iloc[:,i+1],epoch_labs[j+1].iloc[:,i+1])
                
                clus_meas_ari = cluster_measures(layer_sim_meas_ari, epochs=epochs)
                mean_ari = clus_meas_ari.mean_layer_meas_no_out()
                clus_meas_nmi = cluster_measures(layer_sim_meas_nmi, epochs=epochs)
                mean_nmi = clus_meas_nmi.mean_layer_meas_no_out()
                
                #Get best epoch from moving block approach
                best_epoch_ari = clus_meas_ari.moving_block_sdmean(mean_ari)
                #best_epoch_ari = clus_meas_ari.moving_block_meansd(mean_ari)
                epoch_dic_ari[clus_method]=str(best_epoch_ari)
                best_epoch_nmi = clus_meas_nmi.moving_block_sdmean(mean_nmi)
                #best_epoch_nmi = clus_meas_nmi.moving_block_meansd(mean_nmi)
                epoch_dic_nmi[clus_method]=str(best_epoch_nmi)
                
                #Heatmap ARI
                plt.figure(figsize=(8,6), dpi=120)
                ax1 = plt.axes()
                sn.heatmap(layer_sim_meas_ari)
                ax1.add_patch(Rectangle((best_epoch_ari,0),1,len(model.layers), fill=False, edgecolor='black'))
                plt.title("ARI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                plt.ylabel('Layer')
                plt.xlabel('Epoch')
                plt.savefig(epoch_dir+"/"+dataset+"_heatmap_ari.png",bbox_inches='tight')
                
                #Heatmap NMI
                plt.figure(figsize=(8,6), dpi=120) 
                ax1 = plt.axes()
                sn.heatmap(layer_sim_meas_nmi)
                ax1.add_patch(Rectangle((best_epoch_nmi,0),1,len(model.layers), fill=False, edgecolor='black'))
                plt.title("NMI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                plt.ylabel('Layer')
                plt.xlabel('Epoch')
                plt.savefig(epoch_dir+"/"+dataset+"_heatmap_nmi.png",bbox_inches='tight')
                
                 #Plot ARI
                legends = []
                plt.figure(figsize=(8,6), dpi=120)
                ax1 = plt.axes()
                ax1.set_ylim([0,1.01])
                for i in range(len(layer_sim_meas_ari)):
                    legends.append('Layer '+str(i))
                    plt.plot(layer_sim_meas_ari[i], color=cols[i])
                plt.plot(mean_ari, color="black")
                legends.append("mean")
                plt.axvline(x = best_epoch_ari, color = 'black')
                plt.text(best_epoch_ari, 0.055, str(best_epoch_ari), color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
                plt.title("ARI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                plt.ylabel('ARI')
                plt.xlabel('Epoch')
                plt.legend(legends, bbox_to_anchor=(1, 1.025), loc='upper left')
                plt.savefig(epoch_dir+"/"+dataset+"_ari.png",bbox_inches='tight')
                
                #Plot NMI
                legends = []
                plt.figure(figsize=(8,6), dpi=120)
                ax1 = plt.axes()
                ax1.set_ylim([0,1.01])
                for i in range(len(layer_sim_meas_nmi)):
                    legends.append('Layer '+str(i))
                    plt.plot(layer_sim_meas_nmi[i], color=cols[i])
                plt.plot(mean_nmi, color="black")
                legends.append("mean")
                plt.axvline(x = best_epoch_nmi, color = 'black')
                plt.text(best_epoch_nmi, 0.055, str(best_epoch_nmi), color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
                plt.title("NMI Between Epochs\nClus. methods: "+clus_method+str(num_clus)+", Dataset: "+dataset)
                plt.ylabel('NMI')
                plt.xlabel('Epoch')
                plt.legend(legends, bbox_to_anchor=(1, 1.025), loc='upper left')
                plt.savefig(epoch_dir+"/"+dataset+"_nmi.png",bbox_inches='tight')
                
              
                    
        #-----------------------------------------------------------------------------
        #---------------- Plot of Accuracy and loss incl. best epochs ----------------
        #-----------------------------------------------------------------------------

        # ARI - summarize history for accuracy
        plt.figure(figsize=(8,6), dpi=120)
        ax1 = plt.axes()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(test_pd["accuracy"])
        plotted=[]
        for method in epoch_dic_ari:
            plt.axvline(x = int(epoch_dic_ari[method]), color = 'black')
            countt = 0
            for i in range(5):
                countt = countt + plotted.count(str(int(epoch_dic_ari[method])+i))
                countt = countt + plotted.count(str(int(epoch_dic_ari[method])-i))
            if countt>0:
                plt.text(int(epoch_dic_ari[method]), 0.175+0.4*countt, method, color='black', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_ari[method]), 0.055+0.075*countt, epoch_dic_ari[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            else:
                plt.text(int(epoch_dic_ari[method]), 0.175, method, color='black', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_ari[method]), 0.055, epoch_dic_ari[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            plotted.append(epoch_dic_ari[method])
        plt.title("Model Accuracy w. ARI epochs\n Dataset: "+dataset)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_accuracy_epoch_mark_ARI.png")
        plt.show()
        
        # ARI - summarize history for loss
        plt.figure(figsize=(8,6), dpi=120)
        ax1 = plt.axes()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(test_pd["loss"])
        plotted=[]
        for method in epoch_dic_ari:
            plt.axvline(x = int(epoch_dic_ari[method]), color = 'black')
            countt = 0
            for i in range(5):
                countt = countt + plotted.count(str(int(epoch_dic_ari[method])+i))
                countt = countt + plotted.count(str(int(epoch_dic_ari[method])-i))
            if countt>0:
                plt.text(int(epoch_dic_ari[method]), 0.95-0.3*countt, method, color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_ari[method]), 0.055+0.075*countt, epoch_dic_ari[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            else:
                plt.text(int(epoch_dic_ari[method]), 0.95, method, color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_ari[method]), 0.055, epoch_dic_ari[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            plotted.append(epoch_dic_ari[method])
        plt.title("Model Loss w. ARI epochs\n Dataset: "+dataset)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_loss_epoch_mark_ARI.png")
        plt.show()
        
        # NMI - summarize history for accuracy
        plt.figure(figsize=(8,6), dpi=120)
        ax1 = plt.axes()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(test_pd["accuracy"])
        plotted=[]
        for method in epoch_dic_nmi:
            plt.axvline(x = int(epoch_dic_nmi[method]), color = 'black')
            countt = 0
            for i in range(5):
                countt = countt + plotted.count(str(int(epoch_dic_nmi[method])+i))
                countt = countt + plotted.count(str(int(epoch_dic_nmi[method])-i))
            if countt>0:
                plt.text(int(epoch_dic_nmi[method]), 0.175+0.2*countt, method, color='black', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_nmi[method]), 0.055+0.075*countt, epoch_dic_nmi[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            else:
                plt.text(int(epoch_dic_nmi[method]), 0.175, method, color='black', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_nmi[method]), 0.055, epoch_dic_nmi[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            plotted.append(epoch_dic_nmi[method])
        plt.title("Model Accuracy w. NMI epochs\n Dataset: "+dataset)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_accuracy_epoch_mark_NMI.png")
        plt.show()
        
        # NMI - summarize history for loss
        plt.figure(figsize=(8,6), dpi=120)
        ax1 = plt.axes()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(test_pd["loss"])
        plotted=[]
        for method in epoch_dic_nmi:
            plt.axvline(x = int(epoch_dic_nmi[method]), color = 'black')
            countt = 0
            for i in range(5):
                countt = countt + plotted.count(str(int(epoch_dic_nmi[method])+i))
                countt = countt + plotted.count(str(int(epoch_dic_nmi[method])-i))
            if countt>0:
                plt.text(int(epoch_dic_nmi[method]), 0.95-0.2*countt, method, color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_nmi[method]), 0.055+0.075*countt, epoch_dic_nmi[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            else:
                plt.text(int(epoch_dic_nmi[method]), 0.95, method, color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white", rotation = 90)
                plt.text(int(epoch_dic_nmi[method]), 0.055, epoch_dic_nmi[method], color='black', ha='center', va='top', transform=ax1.get_xaxis_transform(), backgroundcolor="white")
            plotted.append(epoch_dic_nmi[method])
        plt.title("Model Loss w. NMI epochs\n Dataset: "+dataset)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(data_dir+"/"+dataset+"_model_loss_epoch_mark_NMI.png")
        plt.show()
        