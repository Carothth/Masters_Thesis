*This README.txt file was generated on 2023-03-22 by Caroline Thingholm 
Thorarinsson*

General Information
------------------
### Title:
*Data and Code for the master's thesis "Clustering Based Explanations of 
Decision Structures in Deep Neural Networks"*

### Author Information:
- Author: Caroline Thingholm Thorarinsson

### Date of creation of project: 
2022-02-01 until 2023-04-01

### Geographic location of data collection:
Department of Mathematics and Computer Science, Univercity of Southern 
Denmark

File & Data Overview
--------------------
### File list:
- `classes.py`
- `main.py`

### Data list:
- `data/mammographic_masses.data`
- `data/mhr.csv`
- `data/obesity.csv`

### Additional notes:
- The analysis is made in `main.py`. This files imports classes from 
  `classes.py`.
- The datasets are all found on https://archive.ics.uci.edu/ml/index.php.
- The analysis is made using Python v. 3.8 and the IDE Spyder v. 4.2.5.

File/Format-Specific Information
--------------------------------
### `classes.py`
This file is developed for the master's thesis "Clustering Based 
Explanations of Decisions Structures in Deep Neural Networks". It contains 
classes for handeling tasks related to inspection of feed forward neural 
networks(FNN's).
                        
The first class 'layer_outputs' contains one function, returning the 
intermediate outputs of a FNN, i.e. the output of each layer of the network
after training.

The second class 'CustomCallback' contains a function returning a 
customized callback to be used in the training of the network if one want
to have the data from each layer after each epoch during training.

The third class 'clustering_layers' contains a function returning a 
clustering of each layer in a given FNN. Different clustering algorithms 
can be chosen.

The fourth class 'layer_plots' contains two functions. The first 'pc_plots' 
constructs a figure with parallel coordinate plots of each of the layers 
after training og the FNN. The second 'pcplot_cluster_layers' constructs a 
figure with parallel coordinate plots for each cluster of each layer after 
training of the FNN.

The fifth class 'epoch_plots' contains one function 'pcplot' returning a 
lis of figure objects. Each of them is a figure with parallel coordinate 
plots of each of the layers efter each epoch during training of the FNN.

The sixth class 'cluster_measures' contains five functions. The first two 
'mean_layer_meas' and 'mean_layer_meas_no_out' computing the mean of a 
similarity measure across the layers, with and without the output layer 
respectively. The last three functions 'moving_block_mean', 
'moving_block_meansd' and 'moving_block_sdmean' finds an estimate of the 
optimal epoch according to a moving block approach. The three functions 
finds the optimal epoch by using only the mean, using first the mean and 
then the standard deviation and first the standard deviation and then the 
mean repectively.

### `main.py`
This file is developed for the master's thesis "Clustering Based Expla-
nations of Decisions Structures in Deep Neural Networks". It contains the
main code for the data management, the network construction, the extraction
of weigts from the networks and the analyses of those. The code is con-
structed in a way such that it is automated to run for all datasets and 
all clustring methods in one run. 
 
This file uses functions from classes defined in the file 'classes.py'.

### `data/mhr.csv`
- Number of variables: 7
- Number of rows: 1014
- Missing data codes: None

### `data/mammographic_masses.data`
- Number of variables: 6
- Number of rows: 961
- Missing data codes: "?"

### `data/obesity.csv`
- Number of variables: 17
- Number of rows: 2111
- Missing data codes: None

Required Packages
--------------------------
### `classes.py`
- os 
- numpy
- pandas
- hdbscan
- sklearn
- matplotlib
- tensorflow.keras
- pickle

### `main.py`
- os
- numpy
- pandas
- matplotlib
- sklearn
- tensorflow.keras
- scipy
- moviepy
- pickle
- seaborn

Costum classes:
- classes

Methodological Information
--------------------------
For description of Neural Networks in general and the clusetering methods 
used in this code see the related master's thesis.
