#!/usr/bin/env python

import pandas as pd
import numpy as np
import re,copy
from collections import defaultdict
import sklearn
from sklearn.linear_model import LogisticRegression
from  sklearn import cross_validation
import matplotlib
import matplotlib.cm as cm 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


lett_to_num = {'A':1,'T':2, 'G':3, "C":4}
num_to_lett = {1:'A',2:'T', 3:'G', 4:"C", -9:"X"}

## Data containers
class TrainData(object):
    """Class for storing per contig train data"""
    def __init__(self, X, y, pos, contig, sample_names=None):
        self.X = X
        self.y = y
        self.pos = pos
        self.contig = contig
        self.sample_names = sample_names
        
class TestData(object):
    """Class for storing per contig test data"""
    def __init__(self, X, pos, contig, truth=None, sample_names=None):
        self.X = X
        self.pos = pos
        self.contig = contig    
        self.truth = truth
        self.sample_names = sample_names
        
class ResultData(object):
    """Class for storing the per contig results, weak ref'ed to the input data"""
    def __init__(self, contig, train=None, test=None, cv=None, classifier=None, p0=None):
        
        self.train = train
        self.test = test
        self.cv = cv
        self.classifier = classifier
        self.p0 = p0
        
    @property
    def pred(self):
        try:
            return self.__pred
        except AttributeError:
            self.__pred = np.empty_like(self.p0, dtype=np.object)
            
            for i in range(self.p0.shape[0]):
                if self.p0[i] > 0.9:
                    self.__pred[i] = 'popB'
                elif self.p0[i] < 0.1:
                    self.__pred[i] = 'popA'
                else:
                    self.__pred[i] = 'X'
            return self.__pred
                

        
        



## Preprocessing
def load_data(markersA_file, markersB_file, markersAB_file, ref_file):
    """ Loads SNP data for the two training groups (A, B) and the test group (AB),
        loads the reference base and constructs a lookup dictionary.
    
        Params:
        markersA_file, markersB_file: csv encoded snps for the two ancestral groups
        markersAB_file: csv encoded snps for the decedent 
        ref_file: file for locus\treference base
    
        Returns:
        markersA, markersB, markersAB :  filtered dataframe reads of the csvs
        ref: dictionary mapping locus to reference base
    """
    
    markersA = pd.read_csv(markersA_file,   comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    markersB = pd.read_csv(markersB_file,   comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    markersAB = pd.read_csv(markersAB_file, comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    
    reference = pd.read_csv(ref_file, comment='#', sep='\t', skip_blank_lines=True)
    ref = { x[0]:str(lett_to_num[x[1]]) for x in reference.to_dict(orient='split')['data']}
    
    assert np.all(markersA['markers'] == markersB['markers']), "The set of loci must be consistent between the groups A, B, AB"
    assert np.all(markersA['markers'] == markersAB['markers']), "The set of loci must be consistent between the groups A, B, AB"
    
    loci = markersA['markers']
    
    return markersA, markersB, markersAB, ref, loci



def encode(marker_df, ref, enc_function):
    """ Encodes the dataframe of bases at each locus per library under the encoding scheme given in enc_function
    
        Params:
        marker_df : dataframe of SNPs for each library from Vanessa
        ref: dictionary mapping contig_pos -> {1,2,3,4} = {A, T, G, C}
        enc_function: function to encode the observed bases relative to the reference args = (ref='2', obs='21')
    
        Returns:
        encoded: np array with shape = (n_loci, n_libraries)
    """
    
    encoded = np.empty((marker_df.shape[0], marker_df.shape[1]-1), dtype=np.object)
    
    for i,line in enumerate(marker_df.iterrows()):
        data = np.array(line[1][1:])
        row = [enc_function(ref[line[1][0]], str(x)) for x in data]
        encoded[i] = row
        
    return encoded

def contig_bunch(marker, loci):
    """ Breaks up the array of (loci, libraries) into a separate contigs
    
        Params:
        marker: output from encode, array (n_loci, n_libraries) 
        loci: list of contig_pos that corresponds with axis 0 of marker 
    
        Returns:
        bunched: dictionary {contig : {'pos':[positions of markers on contig], 'data':array shape (n_pos, n_libraries)}}
    """
    n_loci = len(loci)
    bunched = defaultdict(lambda : {'pos':[], 'data':[]})
    for i in range(n_loci):
        contig,pos = loci[i].split('_')
        bunched[contig]['pos'].append(pos)
        bunched[contig]['data'].append(marker[i])
    
    for contig in bunched.values():
        contig['data'] = np.array(contig['data'])
        
    return bunched

def merge_bunched(bunched1, bunched0):
    """ Merges data bunched by contig of the two groups, creates X and y members.
        NB transposes X array so it is (n_libraries, n_markers)
        Params:
        bunched1: dictionary {contig : {'pos':[positions of markers on contig], 'data':array shape (n_pos, n_libraries)}}
                 y=1
        bunched0: dictionary {contig : {'pos':[positions of markers on contig], 'data':array shape (n_pos, n_libraries)}}
             y=0
    Returns:
        contig_X: {contig : {'pos':[positions of markers on contig], 'X':array shape (n_libraries1+n_libraries0, n_pos), 'y': array shape (n_libraries1+n_libraries0)}}
    """
    
    train_data = {}
    
    for c in bunched0.keys():
        assert bunched1[c]['pos'] == bunched0[c]['pos']
        
        train_data[c] = TrainData(X = np.transpose(np.concatenate(( bunched1[c]['data'], bunched0[c]['data']), axis=1)),
                                  y = np.array([1]*bunched1[c]['data'].shape[1] + [0]*bunched0[c]['data'].shape[1]),
                                  pos = bunched0[c]['pos'],
                                  contig = c)
    return train_data

def create_training_data(markersA, markersB, encode_func, loci, ref):
    """Basically a wrapper that combines the data preprocessing stages for the pipeline for the ancestral populations
        markersA = 1
        markersB = 0
    Params:
    markersA, markersB : dataframes of the snp data for the ancestral populations
    encode_func: function taking (ref_base, obs) to the feature used to train the classifiers
    loci: list of loci names as "conitg_pos"
    ref: dictionary mapping loci to ref_base
    
    Returns:
    training_data: dictionary of training data with keyed by contig: {contig : { 'X':matrix of enc snps, 'y': [1,1,1,...,0,0,0,...], 'pos':[physical pos of snp on contig]}}  
    """
    data = merge_bunched(contig_bunch(encode(markersA, ref, encode_func), loci), 
                         contig_bunch(encode(markersB, ref, encode_func), loci))
    for val in data.values():
        val.sample_names = list(markersA.columns[1:]) + list(markersB.columns[1:])
                                    
    return data
                                      
                                      
def create_test_data(markersAB, encode_func, loci, ref):
    """Basically a wrapper that combines the data preprocessing stages for the pipeline for the descendent population
    
    Params:
    markersAB: dataframe of the snp data for the descendent population
    encode_func: function taking (ref_base, obs) to the feature used to train the classifiers
    loci: list of loci names as "conitg_pos"
    ref: dictionary mapping loci to ref_base
        
    Returns:
    test_data: dictionary of training data with keyed by contig: {contig : { 'X':matrix of enc snps, 'pos':[physical pos of snp on contig]}}  """
    
    bunched = contig_bunch(encode(markersAB, ref, encode_func), loci)
    
    test_data = {}
    for c in bunched.keys():
        test_data[c] = TestData(X = np.transpose(bunched[c]['data']),
                                pos = bunched[c]['pos'],
                                contig = c,
                                sample_names = list(markersAB.columns[1:]))
    return test_data




## Core pipeline
class HyPred(object):
    """docstring for HyPred"""
    def __init__(self, train_data=None, test_data=None, C=1.0, acc_cutoff=0.95, cv_folds=10, classifier=None):
        self.train_data = train_data
        self.test_data = test_data
        self.C = C
        self.acc_cutoff = acc_cutoff
        self.cv_folds = cv_folds
        self.classifier = classifier
        
        
        if self.classifier is None:
            self.classifier = lambda : LogisticRegression(class_weight='balanced', penalty='l1', C=self.C, fit_intercept=False)
    
    def train(self):
        
        """For each contig in train_data train a classifier and do cv_fold cross validation and select 
            contigs which can be classified with accuracy greater than acc_cutoff. Re-train on these 
            selected contigs with all of the training data and update train_data with the trained classifiers
        
            Params:
            train_data : output of create_training_data
            C: L1 regularisation parameter for the default LogisticRegression classifier
            acc_cutoff : Minimum cross validation accuracy for selection
            cv_folds: Number of cross validation folds to use, cannot exceed number of samples in the smallest ancestral population
            classifier : classifier factory to use, defaults to  LogisticRegression(class_weight='balanced', penalty='l1',C=C, fit_intercept=False) 
        
            Returns:
            
            train_data: updated with trained classifiers 'classifier' and their cross validated accuracy 'cv' 
            cv_selected_contigs: list of selected contig names"""
        
                
        self.results_data = {}
        self.selected_contigs = []
        
        for contig in self.train_data.keys():    
    
            cv = cross_validation.cross_val_score(self.classifier(), 
                                                  X=self.train_data[contig].X, 
                                                  y=self.train_data[contig].y, 
                                                  cv=cross_validation.StratifiedKFold(self.train_data[contig].y, self.cv_folds, shuffle=True))
                                                  
            ## Require self.CV accuracy of >self.acc_cutoff to use to evaluate hybridisation 
            if np.mean(cv) > self.acc_cutoff:
                self.selected_contigs.append(contig)
                self.results_data[contig] = ResultData(train=self.train_data[contig],
                                                  cv=cv,
                                                  contig=contig,   
                                                  classifier = self.classifier())
                self.results_data[contig].classifier.fit(X=self.train_data[contig].X, y=self.train_data[contig].y)
                
        print("Successfully trained %i classifiers" % len(self.selected_contigs))#
        
        return len(self.selected_contigs)
        
    def predict(self):
        """Predict ancestral population of origin, using test_data and the trained
         classifiers in results data, for the contigs in selected_contigs
        
        Params:
        results_data: dictionary {contig : ResultData} where the ResultData object has a classifier member populated by a trained classifier """
        
        self.pred_matrix = []
        self.p0_matrix = []
        for contig in self.selected_contigs:
            self.results_data[contig].test = self.test_data[contig]
            self.results_data[contig].p0 = np.array([x[0] for x in self.results_data[contig].classifier.predict_proba(self.test_data[contig].X)])
            
            self.pred_matrix.append(self.results_data[contig].pred)
            self.p0_matrix.append(self.results_data[contig].p0)
            
        self.pred_matrix = np.array(self.pred_matrix)
        self.p0_matrix = np.array(self.p0_matrix)
        
        
    def plot(self):
        """Plot the distribution of ancestral probabilities"""
        sample_names = next(iter(self.test_data.values())).sample_names
        plt.figure()
        for i in range(self.p0_matrix.shape[1]):
            plt.hist(self.p0_matrix[:,i], bins=50, histtype='step', label=sample_names[i])
            plt.xlabel("Pr(Group B)")
        plt.legend()
        plt.show()
    
    def summarise(self):
        """Summary table of ancestral predictions"""
        sample_names = next(iter(self.test_data.values())).sample_names
        
        A = np.sum(self.pred_matrix == "popA", axis=0)
        B = np.sum(self.pred_matrix == "popB", axis=0)
        X = np.sum(self.pred_matrix == "X", axis=0)
        ratio = A/B
        
        print("Sample  \tpopA\tpopB\tno pred\tratio")
        print("-"*60)
        for i in range(len(sample_names)):
            print("{0}\t{1}\t{2}\t{3}\t{4:.2f}".format(sample_names[i], A[i], B[i], X[i], ratio[i]))
    

    def examine_contig(self, contig=None):
        if contig is None:
            contig = next(iter(self.results_data.keys()))
            
        print("Contig {0}\n".format(contig))
        
        print("Sample  \tPrediction\t Pr(B)")
        print("-"*40)
        for s,p, p0 in zip(self.test_data[contig].sample_names, self.results_data[contig].pred, self.results_data[contig].p0  ):
            print("{0}\t{1}\t{2:.2f}".format(s,p,p0))
            
        print("\nCV Accuracy: {0:.2f}".format(np.mean(self.results_data[contig].cv)))
        
        plt.matshow(np.array(self.train_data[contig].X, dtype=np.float), vmax=1, vmin=-1, cmap=cm.seismic)
        plt.yticks(range(len(self.train_data[contig].sample_names)), self.train_data[contig].sample_names, size=4)
        plt.xticks(range(len(self.train_data[contig].pos)), self.train_data[contig].pos, rotation=90, size=4)
        plt.title("Training data")
        
        plt.matshow(np.array(self.test_data[contig].X, dtype=np.float), vmax=1, vmin=-1, cmap=cm.seismic)
        plt.yticks(range(len(self.test_data[contig].sample_names)), self.test_data[contig].sample_names, size=4)
        plt.xticks(range(len(self.test_data[contig].pos)), self.test_data[contig].pos, rotation=90, size=4)
        plt.title("Test data")
        
        plt.matshow(np.array(self.results_data[contig].classifier.coef_, dtype=np.float), 
                            vmax=abs(self.results_data[contig].classifier.coef_).max(), 
                            vmin=-abs(self.results_data[contig].classifier.coef_).max(), 
                            cmap=cm.seismic)
        plt.xticks(range(len(self.test_data[contig].pos)), self.test_data[contig].pos, rotation=90, size=4)
        plt.title("Weight matrix")
    


