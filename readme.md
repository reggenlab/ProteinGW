"# ProteinGW version" 
ProteinGW :
A graph-wavelet based method to model biophysical properties of proteins by computing physicochemical signal of amino-acid in protein residue network. 
ProteinGW also provides scores of influence of each amino-acids on the biophysical property being modelled. This estimate of influence of amino-acid on 
the biophysical-property of a protein helps get insight of effect of missense mutations for diseases. Another application of ProteinGW demonstrated is it 
can help in measuring possible effect of disease associated mutations on proteins using protein property. 

One needs to have python 3.0+, python2.0+ installed in their machine. Following are the dependencies of the code :
Bio
scipy
pygsp
networkx
sklearn
matplotlib
numoy, pandas


You have to download python code in your local machine/server. For execution you have to pass parameters for each choice of task to perform.
ProteinGW has 3 choices of task that can be performed :
1. Modelling Protein Properties
2. Extracting information/scores of most important features for a property
3. Linking cause of disease causing variant


USE THE FOLLOWING COMMANDS :
```bash
python3 Property_modelling.py -p <property> -m <ML_model> -c <Choice of Task to perform>
e.g.
python3 Property_modelling.py -p alpha-beta -m SVM -c 1
```
Options for Properties (-p) are : "protein-folding-rate" , "solubility" , "transmembrane-globular" , "alpha-beta"
Here, protein-folding-rate is a regression task while Solubility, transmembrane-globular, alpha-beta are classfication models.

Options for ML_model (-m) are : 
Classification task (Solubility, transmembrane-globular, alpha-beta) : 
'random-forest' , 'SVM' , 'logistic' , 'naive-bayes' , 'KNN' , 'adaboost'

Regression task (protein-folding-rate) : 
'random-forest' , 'linear' , 'ridge' , 'lasso' , 'decision-tree' , 'KNN' , 'elastic-net'

Options for choice (-c) are : '1', '2', '3'

For task 1 Following parameters are : -p <property> , -m <ML_model> -c 1
For task 2 following parameters are : -p <property> , -c 2
For task 3 following parameters are : -p <property> , -c 3 (-p options are : "protein-folding-rate" , "transmembrane-globular" ) 

