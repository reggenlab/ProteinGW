from __future__ import print_function
from __future__ import print_function
from Bio.PDB import *
import os, collections, subprocess, sys, getopt, traceback, random   
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from pygsp import graphs, features
import networkx as nx
import matplotlib.pyplot as plt
from pygsp import utils, graphs, filters
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.linear_model import LinearRegression , Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics  
from sklearn.metrics import roc_auc_score
from scipy import stats
import traceback
import warnings
warnings.filterwarnings("ignore")

amino_lookup = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
	 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
	 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
	 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M','CCS':'C','AC5':'L'}
amino_molecular_mass = {'A': 89.09404, 'R': 174.20274, 'N': 132.11904, 'D': 133.10384, 'C': 121.15404,
						'Q': 146.14594, 'E': 147.13074, 'G': 75.06714, 'H': 155.15634, 'I': 131.17464,
						'L': 131.17464, 'K': 146.18934, 'M': 149.20784, 'F': 165.19184, 'P': 115.13194,
						'S': 105.09344, 'T': 119.12034, 'W': 204.22844, 'Y': 181.19124, 'V': 117.14784}
amino_hydrophobicity = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
						'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
						'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
						'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

amino_bulkiness = {'A':11.5, 'D':11.68,'C':13.46,'E':13.57,'F':19.8,'G':3.4,'H':13.67,'I':21.4,'K':15.71,'L':21.4,
				   'M':16.25,'N':12.82,'P':17.43,'Q':14.45,'R':14.28,'S':9.47,'T':15.77,'V':21.57,'W':21.61,'Y':18.03}

amino_polarity = {'A':0, 'D':49.7,'C':1.48,'E':49.9,'F':0.35,'G':0,'H':51.6,'I':0.1,'K':49.5,'L':0.13,
				   'M':1.43,'N':3.38,'P':1.58,'Q':3.53,'R':52,'S':1.67,'T':1.66,'V':0.13,'W':2.1,'Y':1.61}

amino_turn_tendency = {'A':0.66, 'D':1.46,'C':1.19,'E':0.74,'F':0.6,'G':1.56,'H':0.95,'I':0.47,'K':1.01,'L':0.59,
				   'M':0.6,'N':1.56,'P':1.52,'Q':0.98,'R':0.95,'S':1.43,'T':0.96,'V':0.5,'W':0.96,'Y':1.14}

amino_coil_tendency = {'A':0.71, 'D':1.21,'C':1.19,'E':0.84,'F':0.71,'G':1.52,'H':1.07,'I':0.66,'K':0.99,'L':0.69,
				   'M':0.59,'N':1.37,'P':1.61,'Q':0.87,'R':1.07,'S':1.34,'T':1.08,'V':0.63,'W':0.76,'Y':1.07}

amino_flexibility = {'A':0, 'D':2,'C':1,'E':3,'F':2,'G':0,'H':2,'I':2,'K':4,'L':2,
				   'M':3,'N':2,'P':0,'Q':3,'R':5,'S':1,'T':1,'V':1,'W':2,'Y':2}

amino_partial_specific_volume = {'A':60.46, 'D':73.83,'C':67.7,'E':85.88,'F':121.48,'G':43.25,'H':98.79,
								 'I':107.72,'K':108.5,
								 'L':107.75,'M':105.35,'N':78.01,'P':82.83,'Q':93.9,
								 'R':127.34,'S':60.62,'T':76.83,'V':90.78,'W':143.91,'Y':123.6}

amino_compressibility = {'A':-25.5, 'D':-33.12,'C':-32.82,'E':-36.17,'F':-34.54,'G':-27,'H':-31.84,
						'I':-31.78,'K':-32.4,
						'L':-31.78,'M':-31.18,'N':-30.9,'P':-23.25,'Q':-32.6,
						'R':-26.62,'S':-29.88,'T':-31.23,'V':-30.62,'W':-30.24,'Y':-35.01}

amino_refractive_index = {'A':14.34, 'D':12,'C':35.77,'E':17.26,'F':29.4,'G':0,'H':21.81,
						'I':19.06,'K':21.29,
						'L':18.78,'M':21.64,'N':13.28,'P':10.93,'Q':17.56,
						'R':26.66,'S':6.35,'T':11.01,'V':13.92,'W':42.53,'Y':31.55}



def crawl_pdb(path):
	'''This function reads pdb files and stores their distance matrix and sequence'''
	parser = PDBParser()
	pdb_files = sorted(os.listdir(path))
	pdbinfo_dict = dict()
	for pdb in pdb_files:
		info = dict()
		info[id] = pdb
		structure = parser.get_structure('pdb_file', path  + pdb )
		coordinates = []
		labels = list()
		for model in structure:
			for chain in model:
				for residue in chain:
					try:
						if residue.get_resname() in amino_lookup:
							coordinates.append(residue['CA'].get_coord())
							labels.append(residue.get_resname())
					except KeyError:
						pass
				break  ## working on chain id A only
			break	  ## Working on model id 0 only
		coords = np.asmatrix(coordinates)
		distance_matrix = distance.squareform(distance.pdist(coords))
		info['coords'] = coords
		info['distance_matrix'] = distance_matrix
#		 print(np.unique(labels))
		info['sequence'] = ''.join([amino_lookup[s] for s in labels if s in amino_lookup])
#		 print(info['sequence'])
		pdbinfo_dict[pdb] = info
	return pdbinfo_dict


def crawl_pdb_alphabeta(path):
	'''This funciton reads pdb files and stores there distance matrix and sequence'''
	parser = PDBParser()
	pdb_files = sorted(os.listdir(path))
	pdbinfo_dict = dict()
	for pdb in pdb_files:
		info = dict()
		info[id] = pdb
		structure = parser.get_structure('pdb_file', path + '' + pdb + '/' + pdb.split('_')[1].upper()+'.pdb' ) # + '/' + pdb + '.pdb'
		coordinates = []
		labels = list()
		for model in structure:
			for chain in model:
				for residue in chain:
					try:
						assert residue.get_resname() not in ['HOH', ' CA']
						coordinates.append(residue['CA'].get_coord())
						labels.append(residue.get_resname())

					except :
						pass
				break  ## working on chain id A only
			break	  ## Working on model id 0 only
		coords = np.asmatrix(coordinates)
		distance_matrix = distance.squareform(distance.pdist(coords))
		info['coords'] = coords
		info['distance_matrix'] = distance_matrix
		info['sequence'] = ''.join([amino_lookup[s] for s in labels])
		pdbinfo_dict[pdb] = info
	return pdbinfo_dict


def get_graph(distance_matrix, network_type, rig_cutoff=8, lin_cutoff=12):
	distance_matrix[distance_matrix >= rig_cutoff] = 0
	if network_type == 'rig-boolean':
		distance_matrix[distance_matrix > 0] = 1
	elif network_type == 'weighted-rig':
		for i in range(np.shape(distance_matrix)[0]):
			for j in range(np.shape(distance_matrix)[1]):
				if distance_matrix[i, j] > 0:
					distance_matrix[i, j] = abs(j - i)
	elif network_type == 'weighted-lin':
		for i in range(np.shape(distance_matrix)[0]):
			for j in range(np.shape(distance_matrix)[1]):
				if distance_matrix[i, j] > 0:
					if abs(i - j) >= lin_cutoff or abs(i - j) == 1:
						distance_matrix[i, j] = abs(i - j)
					else:
						distance_matrix[i, j] = 0
	elif network_type == 'lin':
		for i in range(np.shape(distance_matrix)[0]):
			for j in range(np.shape(distance_matrix)[1]):
				if distance_matrix[i, j] > 0:
					if abs(i - j) >= lin_cutoff or abs(i - j) == 1:
						distance_matrix[i, j] = 1
					else:
						distance_matrix[i, j] = 0
	else:
		print('Invalid Choice! ' + network_type)
		return None
#	 print(distance_matrix.shape)
	G = graphs.Graph(distance_matrix)
	G.compute_fourier_basis()
	return G


def get_signal(G, seq, pdb,signal):
	if signal == 'molecular_weight':
		s = np.asarray([amino_molecular_mass[aa] for aa in seq])
	elif signal == 'hydrophobicity':
		s = np.asarray([amino_hydrophobicity[aa] for aa in seq])
	elif signal == 'node_degree':
		s = G.d
	elif signal == 'node_weighted_degree':
		adj = G.W.todense()
		s = np.ravel(adj.sum(axis=0)) / 2
	elif signal == 'avg_adj_degree':
		s = features.compute_avg_adj_deg(G)
		s = np.ravel(s)
	elif signal == 'clustering_coeff':
		N = nx.from_scipy_sparse_matrix(G.W)
		s = nx.clustering(N)
		s = np.asarray(list(s.values()))
	elif signal == 'aaalpha_helix':
		s = eng.aaalpha_helixfasman(seq)
		s = np.array(s._data)
	elif signal == 'residue_count':
		residue_counts = collections.Counter(seq)
		s = np.asarray([residue_counts[s] for s in seq])
	elif signal == 'bulkiness':
		s = np.asarray([amino_bulkiness[aa] for aa in seq])
	elif signal == 'polarity':
		s = np.asarray([amino_polarity[aa] for aa in seq])
	elif signal == 'turn_tendency':
		s = np.asarray([amino_turn_tendency[aa] for aa in seq])
	elif signal == 'coil_tendency':
		s = np.asarray([amino_coil_tendency[aa] for aa in seq])
	elif signal == 'flexibility':
		s = np.asarray([amino_flexibility[aa] for aa in seq])
	elif signal == 'partial_specific_volume':
		s = np.asarray([amino_partial_specific_volume[aa] for aa in seq])
	elif signal == 'compressibility':
		s = np.asarray([amino_compressibility[aa] for aa in seq])
	elif signal == 'refractive_index':
		s = np.asarray([amino_refractive_index[aa] for aa in seq])
	elif signal == 'conservation_score':
		#https://compbio.cs.princeton.edu/conservation/
		filename = pdb.split('.')[0]
		#cmd = ['python3 ./pdb2fasta-master/pdb2fasta.py '+pdb_path+''+pdb+' > ./pdb2fasta-master/'+filename+'.fasta']
		#print(cmd)
		#process = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		with open('./pdb2fasta-master/'+filename+'.fasta', 'w') as the_file:
			the_file.write('>'+filename+':A\n'+seq+"-")
		process = 0
		if process ==0:
			s = []
			cmd = ['python2 ./pdb2fasta-master/conservation_code/score_conservation.py -alignfile ./pdb2fasta-master/'+filename+'.fasta > ./pdb2fasta-master/'+filename+'.csv']
#			 print(cmd)
			process = subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if process ==0:
				with open('./pdb2fasta-master/'+filename+'.csv') as f:
					for i in range(5):
						next(f)
					for line in f:
#						 print(line.split("\t")[1])
						s.append(float(line.split("\t")[1]))
		s = np.array(s)
	else:
		print ('Invalid Choice! ' + signal)
	return s

def get_cutoff(signal,G):
	coeff = []
	cutoff = []
	for i in range(20,90,10):
		p = np.percentile(signal, i) 
#		 print(p)
		signal[np.where(signal<p)] = 0
#		 print(np.corrcoef(signal,G.e))
		coeff.append(np.corrcoef(signal,G.lmax*G.e)[0,1])
		cutoff.append(i)
#	 print("coeff",coeff)
	cutoff = cutoff[np.argmax(coeff)]
	return cutoff

def get_filtered_signal(G, signal, cutoff,type_spatial):
	if type_spatial == 'fourier':
		gftsignal = G.gft(signal)
		signal_hat = gftsignal
		value = np.sum(abs(signal_hat[G.e < G.lmax*cutoff])) / np.sum(abs(signal_hat))
		return value
	elif type_spatial == 'wavelet':
		N_f=4
		scales = utils.compute_log_scales(1, len(signal), N_f-1)
		mex = filters.Abspline(G, Nf=N_f,scales=scales)
#		 for i, signal in enumerate(exp):
		signal_filtered_hat = mex.filter(signal)
		signal_filtered_hat = np.abs(signal_filtered_hat)
		for j in range(signal_filtered_hat.shape[1]):
				cutof= get_cutoff(signal_filtered_hat[:,j],G)
				p=np.percentile(signal_filtered_hat[:,j],cutoff) 
				signal_filtered_hat[np.where(signal_filtered_hat[:,j]<p),j] = 0
#		 inv_fil = mex.filter(signal_filtered_hat)
#		 print(inv_fil.shape)
		signal_filtered_hat = np.mean(np.abs(signal_filtered_hat),axis=0)
		return signal_filtered_hat


def crawl_pdb_solubility(path):
	'''This funciton reads pdb files and stores there distance matrix and sequence'''
	parser = PDBParser()
	pdb_files = sorted(os.listdir(path))
	pdbinfo_dict = dict()
#	 print(pdb_files)
	for pdb in pdb_files:
		info = dict()
#		 print(path + pdb)
		structure = parser.get_structure('pdb_file', path + pdb ) # + '/' + pdb + '.pdb'
		coordinates = []
		labels = list()
		for model in structure:
			for chain in model:
				for residue in chain:
					try:
						assert residue.get_resname() not in ['HOH', ' CA']
						coordinates.append(residue['CA'].get_coord())
						labels.append(residue.get_resname())

					except :
						pass
				break  ## working on chain id A only
			break	  ## Working on model id 0 only
		coords = np.asmatrix(coordinates)
		distance_matrix = distance.squareform(distance.pdist(coords))
		try:
			info[id] = pdb
			info['coords'] = coords
			info['distance_matrix'] = distance_matrix
			info['sequence'] = ''.join([amino_lookup[s] for s in labels])
			pdbinfo_dict[pdb] = info
		except:
			pass
	return pdbinfo_dict

def crawl_pdb_folding(path):
	'''This function reads pdb files and stores their distance matrix and sequence'''
	parser = PDBParser()
	pdb_files = sorted(os.listdir(path))
	pdbinfo_dict = dict()
	for pdb in pdb_files:
		info = dict()
		info[id] = pdb
		structure = parser.get_structure('pdb_file', path  + pdb +"/pdb"+ pdb.lower() +".ent" )
		coordinates = []
		labels = list()
		for model in structure:
			for chain in model:
				for residue in chain:
					try:
						if residue.get_resname() in amino_lookup:
							coordinates.append(residue['CA'].get_coord())
							labels.append(residue.get_resname())
					except KeyError:
						pass
				break  ## working on chain id A only
			break	  ## Working on model id 0 only
		coords = np.asmatrix(coordinates)
		distance_matrix = distance.squareform(distance.pdist(coords))
		info['coords'] = coords
		info['distance_matrix'] = distance_matrix
#		 print(np.unique(labels))
		info['sequence'] = ''.join([amino_lookup[s] for s in labels if s in amino_lookup])
#		 print(info['sequence'])
		pdbinfo_dict[pdb] = info
	return pdbinfo_dict


signals_wavelet = []
signals = ['molecular_weight', 'hydrophobicity', 'node_degree', 'node_weighted_degree', 'residue_count', 'clustering_coeff','conservation_score','bulkiness', 'polarity', 'turn_tendency' , 'coil_tendency' , 'flexibility', 'partial_specific_volume','refractive_index','compressibility']
for i in signals:
	for j in range(1,5):
		signals_wavelet.append(i+"_"+str(j))

def get_filtered_signal_mutation(G, signal, cutoff,type_spatial, indices):
    if type_spatial == 'fourier':
        gftsignal = G.gft(signal)
        signal_hat = gftsignal
        value = np.sum(abs(signal_hat[G.e < G.lmax*cutoff])) / np.sum(abs(signal_hat))
        return value
    elif type_spatial == 'wavelet':
        N_f=4
        scales = utils.compute_log_scales(1, len(signal), N_f-1)
        mex = filters.Abspline(G, Nf=N_f,scales=scales)
        signal_filtered_hat = mex.filter(signal)
        signal_filtered_hat = np.abs(signal_filtered_hat)
        signal_filtered_hat1 = np.zeros([1,signal_filtered_hat.shape[1]])
        for j in range(signal_filtered_hat.shape[1]):
            p = np.percentile(signal_filtered_hat[:,j], 70) 
            signal_filtered_hat[np.where(signal_filtered_hat[:,j]<p),j] = 0        
            b = [stats.percentileofscore(signal_filtered_hat[:,j], a, 'rank') for a in signal_filtered_hat[:,j]]
            signal_filtered_hat[:,j] = b
#             print(signal_filtered_hat[:,j])
            signal_filtered_hat1[:,j] = signal_filtered_hat[indices,j] 
        signal_filtered_hat1 = np.mean(np.abs(signal_filtered_hat1))
        return signal_filtered_hat1

def mutation_find_wavelet_coefficient(dictionary,residue_mutation,amino_lookup,signal_important,cutoff,type_spatial,network_type):
    residue_mutation['pdb'] = [x.upper() for x in residue_mutation['pdb']]
    keys = dictionary.keys()
    d1 = {}
    for i in keys:
        i1 = i.split('.')[0]
        i1 = i1.upper()
        d1[i1] = dictionary[i]  
    G = {}
    for pdb in d1.keys():
        try:
            G[pdb] = get_graph(d1[pdb]['distance_matrix'], network_type=network_type, rig_cutoff=7.3)
        except:
            continue  
    coeff_final = {}
    for k in d1.keys():
#         coeff_final[k] = {}
        indx = np.where(residue_mutation['pdb'] == k)[0]
        residues = residue_mutation.loc[indx,'Residue']
        pos = np.array(residue_mutation.loc[indx,'Position'])
        dis = np.array(residue_mutation.loc[indx,'Disease name'])
        residues = [amino_lookup.get(key.upper()) for key in residues]
        to_mutate = np.array(residue_mutation.loc[indx,'Mutate_to_residue'])
#         residues = np.unique(residues)
        G1 = G[k]
#         print(np.array(list(d1[k]['sequence'])))
        for m,j in enumerate(residues):
#             indices = [l for l,s in enumerate(d1[k]['sequence']) if j in s]
            indices = pos[m]
            try:
                if np.array(list(d1[k]['sequence']))[indices] == j: 
                    print("protein",k)
                    print("residue",j)
                    print("position",pos[m])
                    print("disease",dis[m])
                    print("to_mutate",to_mutate[m])
                    coeff_final[k,j,pos[m],dis[m]] = {}
                    for i in signal_important:
#                         try:
                            coeff_final[k,j,pos[m],dis[m]][i] = {}
                            signal = get_signal(G1, d1[k]['sequence'],k,signal=i)
                            coeff1 = get_filtered_signal_mutation(G1, signal, cutoff, type_spatial, indices)
        #                     print(coeff1)
                            coeff_final[k,j,pos[m],dis[m]][i] = coeff1  
#                         except:
#                             if coeff_final[k][j,pos[m]][i] == {}:
#                                 del coeff_final[k][j,pos[m]][i] 
                    if coeff_final[k,j,pos[m],dis[m]] == {}:
                                del coeff_final[k,j,pos[m],dis[m]]  
            except:
                print("")
    return coeff_final


def main(argv):
	property = ''
	choice = ''
	ML_model = ''
	try:
		opts, args = getopt.getopt(argv,"hc:p:m:")
	except getopt.GetoptError:
		print("Property_modelling.py -p <property> -m <ML_model> -c <Choice of Task to perform>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("Property_modelling.py -c <Choice of Task to perform> -p <property> -m <ML_model>")
			sys.exit()
		elif opt in ("-c"):
			 choice = arg
		elif opt in ("-p"):
			 property = arg
		elif opt in ("-m"):
			ML_model = arg
	if property == '' or ML_model == '' or choice == '':
		print("Choice of Task/ Property to model/ ML_model is required!!")
	else:	
		model = 'weighted-rig'
		gsp_features = pd.DataFrame(columns=signals_wavelet + ['class'])

		if property == 'alpha-beta':
			path = '../Protein-GSP-master/data/SDTSF_RIG_LIN/'
			pdbinfo_dict = crawl_pdb_alphabeta(path)

			for pdb in pdbinfo_dict.keys():
		#		 print (pdb, end=', ')
				row = []
				if pdb.startswith('A_'): c = 1
				elif pdb.startswith('B_'): c = -1
				else: c = 0
				G = get_graph(pdbinfo_dict[pdb]['distance_matrix'], network_type=model, rig_cutoff=7.3)
				for signal_name in signals:
					signal = get_signal(G, pdbinfo_dict[pdb]['sequence'],pdb,signal=signal_name)
					value = get_filtered_signal(G,signal,cutoff=70,type_spatial='wavelet')	   
					row.extend(value)
				row.append(c)
				gsp_features.loc[pdb] = row

			gsp_features = gsp_features.drop(gsp_features.index[[0,1,2,3,4,5,6,7,8,9,10,11]])

			X = gsp_features[gsp_features.columns.difference(['class'])]
			y = gsp_features['class']

			print("Class Assignment :\n")
			print("Alpha = 1, Beta = -1")
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70,random_state=9)
			if choice == '1':
				if ML_model == 'random-forest':
					clf = RandomForestClassifier(n_estimators = 1000)			
				elif ML_model == 'SVM':
					clf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
				elif ML_model =='logistic':
					clf = LogisticRegression()
				elif ML_model =='naive-bayes':		
					clf = GaussianNB()
				elif ML_model=='KNN':
					clf =  KNeighborsClassifier(n_neighbors=5)
				elif ML_model == 'adaboost':
					clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
				clf.fit(X_train, y_train) 
	  
				# performing predictions on the test dataset 
				y_pred = clf.predict(X_test) 
				print() 
				# using metrics module for accuracy calculation 
				#print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 

				# print classification report 
				print(classification_report(y_test, y_pred)) 
				print("AUCROC",roc_auc_score(y_test, clf.predict(X_test)))
				print("MCC :",matthews_corrcoef(y_test, y_pred))

		elif property == 'transmembrane-globular':
			pdbinfo_dict_trans = crawl_pdb('../Protein-GSP-master/data/transmembrane/')
			pdbinfo_dict_glob = crawl_pdb('../Protein-GSP-master/data/globular/')
			pdbinfo_dict = pdbinfo_dict_trans.copy()
			#print(pdbinfo_dict)
			pdbinfo_dict.update(pdbinfo_dict_glob)
			print(pdbinfo_dict.keys())

			G_glob = {}
			G_trans = {}

			for pdb in pdbinfo_dict_glob.keys():
				try:
					G_glob[pdb] = get_graph(pdbinfo_dict_glob[pdb]['distance_matrix'], network_type=model, rig_cutoff=7.3)
				except:
					continue
			for pdb in pdbinfo_dict_trans.keys():
				try:
					G_trans[pdb] = get_graph(pdbinfo_dict_trans[pdb]['distance_matrix'], network_type=model, rig_cutoff=7.3)
				except:
					continue
			gsp_features = pd.DataFrame(columns=signals_wavelet + ['class'])

			for pdb in G_glob.keys():
					row = []
					c = -1
					G = G_glob[pdb]
					if pdbinfo_dict_glob[pdb]['sequence'] != '':
						for signal_name in signals:
								signal = get_signal(G, pdbinfo_dict_glob[pdb]['sequence'],pdb,signal=signal_name)
								value = get_filtered_signal(G,signal,cutoff=70,type_spatial='wavelet')	   
								row.extend(value)
						row.append(c)
						gsp_features.loc[pdb] = row
					else:
						pass
			
			
			for pdb in G_trans.keys():
					row = []
					c = 1
					G = G_trans[pdb]
					if pdbinfo_dict_trans[pdb]['sequence'] != '':
						for signal_name in signals:
								signal = get_signal(G, pdbinfo_dict_trans[pdb]['sequence'],pdb,signal=signal_name)
								value = get_filtered_signal(G,signal,cutoff=70,type_spatial='wavelet')	   
								row.extend(value)
						row.append(c)
						gsp_features.loc[pdb] = row
					else:
						pass
			X = gsp_features[gsp_features.columns.difference(['class'])]
			y = gsp_features['class']

			print("Class Assignment :\n")
			print("Globular = -1, Transmembrane = 1")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
			
			if choice == '1':
				if ML_model == 'random-forest':
					clf = RandomForestClassifier(n_estimators = 1000)			
				elif ML_model == 'SVM':
					clf = SVC(kernel = 'linear',gamma = 'scale', shrinking = False)
				elif ML_model =='logistic':
					clf = LogisticRegression()
				elif ML_model =='naive-bayes':		
					clf = GaussianNB()
				elif ML_model=='KNN':
					clf =  KNeighborsClassifier(n_neighbors=5)
				elif ML_model == 'adaboost':
					clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
				clf.fit(X_train, y_train) 
	  
				# performing predictions on the test dataset 
				y_pred = clf.predict(X_test) 
	  
				# metrics are used to find accuracy or error  
				print()   
				# using metrics module for accuracy calculation 
				#print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
				# print classification report 
				print(classification_report(y_test, y_pred)) 
				print("AUCROC",roc_auc_score(y_test, clf.predict(X_test)))
				print("MCC :",matthews_corrcoef(y_test, y_pred))

		elif property == 'solubility':
			solubility_ground = pd.read_csv("../Protein-GSP-master/data/solubility_percentage_new.txt",sep=",")
			solubility_ground.index = solubility_ground['pdb']
			solubility_ground['Solubility(%)'] = solubility_ground['Solubility(%)'].astype('int')
			path = '../Protein-GSP-master/data/solubility_data/'
			pdbinfo_dict = crawl_pdb_solubility(path)
			#print(pdbinfo_dict)
			gsp_features = pd.DataFrame(columns=signals_wavelet + ['class'])
			for pdb in pdbinfo_dict.keys():
				try:
					#print(pdb)
					#print("here")
					row = []
					c = solubility_ground.loc[pdb,'Solubility(%)']
					#print(c)
					G = get_graph(pdbinfo_dict[pdb]['distance_matrix'], network_type=model, rig_cutoff=7.3)
					for signal_name in signals:
						signal = get_signal(G, pdbinfo_dict[pdb]['sequence'],pdb,signal=signal_name)
						value = get_filtered_signal(G,signal,cutoff=70,type_spatial='wavelet')   
						row.extend(value)
					row.append(int(c))
					#print(row)
					gsp_features.loc[pdb] = row
				except:
					print("")
			print(gsp_features)
			#a_series = (gsp_features != 0).any(axis=1)
			#gsp_features = gsp_features.loc[a_series]
			print(gsp_features.shape)
			for i in range(gsp_features.shape[0]):
				if gsp_features.iloc[i]['class']>64: gsp_features.iloc[i]['class'] = 1
				else: gsp_features.iloc[i]['class'] = 0

			X = gsp_features[gsp_features.columns.difference(['class'])]
			y = gsp_features['class']

			print("Class Assignment :\n")
			print("Soluble = 1, Insoluble = 0")

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
			if choice == '1':
				if ML_model == 'random-forest':
					clf = RandomForestClassifier(n_estimators = 1000)			
				elif ML_model == 'SVM':
					clf = SVC(kernel = 'poly')
				elif ML_model =='logistic':
					clf = LogisticRegression()
				elif ML_model =='naive-bayes':		
					clf = GaussianNB()
				elif ML_model=='KNN':
					clf =  KNeighborsClassifier(n_neighbors=5)
				elif ML_model == 'adaboost':
					clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
				clf.fit(X_train, y_train) 
	  
				# performing predictions on the test dataset 
				y_pred = clf.predict(X_test) 
	  
				# metrics are used to find accuracy or error 
				from sklearn import metrics   
				print() 	  
				# using metrics module for accuracy calculation 
				#print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
				# print classification report 
				print(classification_report(y_test, y_pred)) 
				print("AUCROC",roc_auc_score(y_test, clf.predict(X_test)))
				print("MCC score :",matthews_corrcoef(y_test, y_pred))
				
		elif property == 'protein-folding-rate':
			path = '../Protein-GSP-master/data/regression_model/test_pdb/'
			pdbinfo_dict_1 = crawl_pdb_folding(path)

			pdbinfo_dict = pdbinfo_dict_1.copy()   # start with x's keys and values
			# pdbinfo_dict.update(pdbinfo_dict_2) 

			df = pd.read_csv('../Protein-GSP-master/data/regression_model/final_lnkf.csv', index_col=0)
			lnkfs = df['Ln.K_f.']
			lnkfs = lnkfs[~lnkfs.index.duplicated(keep='first')]

			gsp_features = pd.DataFrame(columns=signals_wavelet + ['class'])

			for pdb in pdbinfo_dict.keys():
					row = []
					c = lnkfs[pdb.upper()]
					G = get_graph(pdbinfo_dict[pdb]['distance_matrix'], network_type=model, rig_cutoff=7.3)
					#print("c :",c)
					#print(pdbinfo_dict[pdb])
					if pdbinfo_dict[pdb]['distance_matrix'].shape[0] == 1:
						pass
					else:
						for signal_name in signals:
								signal = get_signal(G, pdbinfo_dict[pdb]['sequence'],pdb,signal=signal_name)
								#print(signal)
								value = get_filtered_signal(G,signal,cutoff=50,type_spatial='wavelet')	
								row.extend(value)
						row.append(c)
						gsp_features.loc[pdb] = row

			# gsp_features.dropna()
			X = gsp_features[gsp_features.columns.difference(['class'])]
			y = gsp_features['class']
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
			if choice == '1':
				if ML_model == 'random-forest':
					model = RandomForestRegressor(n_estimators=1000)
				elif ML_model == 'linear':
					model = LinearRegression()
				elif ML_model == 'ridge':
					model = Ridge(alpha=0.01)
				elif ML_model == 'lasso':
					model = Lasso(alpha=0.01)
				elif ML_model == 'decision-tree':
					model = DecisionTreeRegressor(random_state=0)
				elif ML_model == 'KNN':
					model = KNeighborsRegressor(n_neighbors=5)
				elif ML_model == 'elastic-net':
					model = ElasticNet(alpha = 0.01)
				
				model.fit(X_train, y_train)
				sc = model.score(X_test,y_test)
				#	 scores.append(sc)
				#	 corr_matrix = numpy.corrcoef(y_test, model.predict(X_test))
				corr_matrix = stats.spearmanr(y_test, model.predict(X_test))
				corr = corr_matrix[0]
				#scores.append(corr)
				rmse = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
				#scores = np.array(scores)
				print("R score: ", corr)

				#rmse = np.array(rmse)
				print("RMSE: ", rmse)


		if choice == '2':
			if property == 'protein-folding-rate':
				model = RandomForestRegressor(n_estimators=1000)
			else:
				model = RandomForestClassifier(n_estimators=1000)
			model.fit(X_train, y_train)
			sc = model.score(X_test,y_test)
			#get importance

			#f, ax = plt.subplots(figsize=(10,3))
			importance = model.feature_importances_
			# summarize feature importance
			for i,v in enumerate(importance):
					print('Feature: %0d, Score: %.5f' % (i,v))
			# plot feature importance
			pyplot.bar([x for x in range(len(importance))], importance)
			y_pos = np.arange(X.shape[1])
			pyplot.xticks(y_pos,signals_wavelet, rotation='vertical')
			plt.savefig('feature_importance.png')
			pyplot.show()
			final_data = pd.DataFrame(columns = ['Wavelet Scales','Feature Score','Feature Importance','Feature'])
			k=0
			for i,j in zip(importance,signals_wavelet):
				final_data.loc[k,'Wavelet Scales'] = j.rpartition("_")[2]
				final_data.loc[k,'Feature'] = j.rpartition("_")[0]
				final_data.loc[k,'Feature Score'] = i
				final_data.loc[k,'Feature Importance'] = i
				k=k+1
			final_data.to_csv("./feature_importance.txt",sep="\t",index=False)
			print("Graph saved to directory with name -- feature_importance.png")
			print("Feature importance scores saved to directory with name -- feature_importance.txt")
		#Disease Residue Scores
		if choice == '3':
			print("")
			model = RandomForestRegressor(n_estimators=1000)
			model.fit(X_train, y_train)
			sc = model.score(X_test,y_test)

			#get importance
			importance = model.feature_importances_
			important_features = X.columns
			important_features = [x.rsplit("_",1)[0] for x in important_features]
			important_features = np.unique(important_features)

			if property == 'protein-folding-rate':
				residue_mutation = pd.read_csv("./protein_folding_rate_residue_mutation.txt",sep="\t")
			elif property == 'transmembrane-globular':
				residue_mutation = pd.read_csv("./trans_glob_residue_mutation.txt",sep="\t")			
			residue_mutation.columns = ['index','pdb','AA_change','Disease name','Residue','Position','Mutate_to_residue',''] 
			cutoff=70
			type_spatial='wavelet'
			network_type = 'weighted-rig'
			final_dict = mutation_find_wavelet_coefficient(pdbinfo_dict,residue_mutation,amino_lookup,important_features,cutoff,type_spatial,network_type)
			file = open('disease_residue_score.txt', 'wt')
			file.write(str(final_dict))
			file.close()
			print(final_dict)
			print('Mutation residue scores saved in file disease_residue_score.txt')
if __name__ == "__main__":
	main(sys.argv[1:])
