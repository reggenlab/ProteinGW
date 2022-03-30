"# ProteinGW version" <br />
<h4>ProteinGW :</h4> <br />
A graph-wavelet based method to model biophysical properties of proteins by computing physicochemical signal of amino-acid in protein residue network. 
ProteinGW also provides scores of influence of each amino-acids on the biophysical property being modelled. This estimate of influence of amino-acid on 
the biophysical-property of a protein helps get insight of effect of missense mutations for diseases. Another application of ProteinGW demonstrated is it 
can help in measuring possible effect of disease associated mutations on proteins using protein property. 
<br />
One needs to have python 3.6+ installed in their machine. Following are the dependencies of the code : <br />
biopython (https://biopython.org/)<br />
scipy <br />
pygsp (https://pygsp.readthedocs.io/en/stable/) <br />
networkx <br />
sklearn <br />
matplotlib <br />
numpy, pandas <br />

One can download the whole folder from our website http://reggen.iiitd.edu.in:1207/ProteinGW/ProteinGW/ <br>
or more information about tool can be accessed from this website.
User can also download the dataset from http://reggen.iiitd.edu.in:1207/ProteinGW/

<br />
You have to download python code in your local machine/server. For execution you have to pass parameters for each choice of task to perform. <br />
ProteinGW has 3 choices of task that can be performed : <br />
1. Modelling Protein Properties <br />
2. Extracting information/scores of most important features for a property <br />
3. Linking cause of disease causing variant <br />

Note : The script Property_modelling.py is in "code" folder. Please cd into the "code" directory to run following command. <br/>
USE THE FOLLOWING COMMANDS : <br />
```bash
python3 Property_modelling.py -p <property> -m <ML_model> -c <Choice of Task to perform>
e.g.
python3 Property_modelling.py -p solubility -m SVM -c 1
```
Options for Properties (-p) are : "protein-folding-rate" , "solubility" , "transmembrane-globular" , "alpha-beta" <br />
Here, protein-folding-rate is a regression task while Solubility, transmembrane-globular, alpha-beta are classfication models. <br />

Options for ML_model (-m) are :  <br />
Classification task (Solubility, transmembrane-globular, alpha-beta) :  <br />
'random-forest' , 'SVM' , 'logistic' , 'naive-bayes' , 'KNN' , 'adaboost' <br />

Regression task (protein-folding-rate) :  <br />
'random-forest' , 'linear' , 'ridge' , 'lasso' , 'decision-tree' , 'KNN' , 'elastic-net' <br />

Options for choice (-c) are : '1' (Modelling Protein Properties), '2' (Extracting information/scores of most important features for a property), '3' (Extracting information/scores of most important features for a property)<br />
For task 1 Following parameters are : -p <property> , -m <ML_model> -c 1  <br />
For task 2 following parameters are : -p <property> , -c 2 <br />
For task 3 following parameters are : -p <property> , -c 3 (-p options are : "protein-folding-rate" , "transmembrane-globular" )  <br />

