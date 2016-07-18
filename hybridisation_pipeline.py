#!/usr/bin/env python

import pandas as pd
import numpy as np
import re,copy,time
from collections import defaultdict
import sklearn
from sklearn.linear_model import LogisticRegression
from  sklearn import cross_validation
import sklearn.preprocessing 
import matplotlib
import matplotlib.cm as cm 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.colors import ListedColormap


lett_to_num = {'A':1,'T':2, 'G':3, "C":4}
num_to_lett = {1:'A',2:'T', 3:'G', 4:"C", -9:"X"}

class Data(object):
    """docstring for Data"""
    def __init__(self, X, pos, contig, sample_names=None):
        super(Data, self).__init__()
        self.X = X
        self.pos = pos
        self.contig = contig
        self.sample_names = sample_names
        self.n_samples, self.n_loci = X.shape
        self.het_X = None
        
        if sample_names is not None:
            assert len(sample_names) == X.shape[0], "Dimensions of sample_names must match X"
        
        

 
 

## Data containers
class TrainData(Data):
    """Class for storing per contig train data"""
    def __init__(self, X, y, pos, contig, sample_names=None):
        super(TrainData, self).__init__(X, pos, contig, sample_names)
        self.y = y

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
    
    
        ## Deep copy everything...
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        
        ## Override deep copy for the arrays to use np.copy, as apparently deepcopy(np.array) isn't working....
        setattr(result, 'X', np.copy(self.X))
        setattr(result, 'y', np.copy(self.y))
        return result
    
class TestData(Data):
    """Class for storing per contig test data"""
    def __init__(self, X, pos, contig, truth=None, sample_names=None, het_X=None):
        super(TestData, self).__init__(X, pos, contig, sample_names)
        self.het_X = het_X
        self.truth = truth
        
        #if self.het_X is not None:
        #    self.growHetTree()
            
    def growHetTree(self):
        """Unpacks the compressed representation of heterokaryotic data to a full tree of possible configurations"""
        self.het_tree = [None for i in range(len(self.sample_names))]
        for sample_i in range(self.het_X.shape[0]):
            het_sites = np.sum([len(x) == 2 for x in self.het_X[sample_i]])
            if het_sites > 16:
                raise Exception("To many hetrokaryotic sites")
            self.het_tree[sample_i] = np.empty((2**het_sites, self.n_loci), dtype=np.int8)
            i = np.arange(2**het_sites)
            
            for locus_i in range(self.n_loci):
                if len(self.het_X[sample_i, locus_i]) == 1:
                    self.het_tree[sample_i][:, locus_i] = self.het_X[sample_i, locus_i][0]
                else:
                    het_sites -= 1
                    ## For each site futher subdivide the block matrix
                    self.het_tree[sample_i][:, locus_i] = np.where(np.floor(i/2**het_sites) % 2, 
                                                                   self.het_X[sample_i, locus_i][0], 
                                                                   self.het_X[sample_i, locus_i][1])
                        

class Result(object):
    """Class for storing the per contig results, weak ref'ed to the input data"""
    def __init__(self, contig, train=None, test=None, cv=None, classifier=None, p0=None):
        
        self.train = train
        self.test = test
        self.cv = cv
        self.classifier = classifier
        self.p0 = p0
        self.contig=contig
        
    @property
    def pred(self):
        self.__pred = np.empty_like(self.p0, dtype=np.object)
        if type(self.p0[0]) is list:
            self.__pred = [map(_pred, x) for x in self.p0]
        else:
            self.__pred = map(_pred, self.p0)
            
        return self.__pred
   
   
   
def _pred(p):
    if p > 0.9:
        return 'popB'
    elif p < 0.1:
        return 'popA'
    else:
        return 'X'

col = ListedColormap(['white', 'red', 'blue', 'green'], 'indexed')

def mp(x):
    if x == [0]:
        return 0
    elif x == [1]:
        return 1
    elif x == [-1]:
        return 2
    elif x == [-1,1]:
        return 3

## Preprocessing
def load_data(markersA_file, markersB_file, markersAB_file, ref_file, contig_data_file=None):
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
    
    if contig_data_file is not None:
        contig_data = pd.read_csv(contig_data_file, comment='#', skip_blank_lines=True, index_col=0)
        return markersA, markersB, markersAB, ref, loci, contig_data
    
    else:
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
    stats = {'Homokaryotic reference':0, 'Heterokaryotic reference/variant':0, 'Homokaryotic variant':0, 'Heterokaryotic variant':0, 'missing':0}
    
    for i,line in enumerate(marker_df.iterrows()):
        data = np.array(line[1][1:])
        row = [enc_function(ref[line[1][0]], str(x), stats) for x in data]
        encoded[i] = row
    
    print("-"*60)
        
    for k,v in stats.items():
        print("{0}: {1}".format(k,v))
        
    cov = float(stats['Homokaryotic reference'] + stats['Heterokaryotic reference/variant'] + stats['Homokaryotic variant'] + stats['Heterokaryotic variant'])
    
    print("\nHomo: {0:.2f}%  Hetero: {1:.2f}%".format(100*(stats['Homokaryotic reference']+stats['Homokaryotic variant'])/cov, 
                                                    100*(stats['Heterokaryotic reference/variant']+stats['Heterokaryotic variant']) /cov ) )
                                                    
    print("Covered: {0:.2f}%  Missing: {1:.2f}%".format(100*cov/(stats['missing']+cov),
                                                       100*stats['missing']/(stats['missing']+cov) )) 
                                                       
    print("-"*60)
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

def create_training_data(markersA, markersB, encode_func, loci, ref, contig_data=None):
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
    
    for key,val in data.items():
        val.sample_names = list(markersA.columns[1:]) + list(markersB.columns[1:])
        if contig_data is not None:
             val.contig_data = contig_data.ix[int(key)].to_dict()
             
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

def create_test_data_heterokaryotic(markersAB, hom_encode_func, het_encode_func, loci, ref):
    """Basically a wrapper that combines the data preprocessing stages for the pipeline for the descendent population
    
    Params:
    markersAB: dataframe of the snp data for the descendent population
    encode_func: function taking (ref_base, obs) to the feature used to train the classifiers
    loci: list of loci names as "conitg_pos"
    ref: dictionary mapping loci to ref_base
        
    Returns:
    test_data: dictionary of training data with keyed by contig: {contig : { 'X':matrix of enc snps, 'pos':[physical pos of snp on contig]}}  """
    
    hom_bunched = contig_bunch(encode(markersAB, ref, hom_encode_func), loci)
    het_bunched = contig_bunch(encode(markersAB, ref, het_encode_func), loci)
    
    test_data = {}
    for c in hom_bunched.keys():
        test_data[c] = TestData(X = np.transpose(hom_bunched[c]['data']),
                                het_X = np.transpose(het_bunched[c]['data']),
                                pos = hom_bunched[c]['pos'],
                                contig = c,
                                sample_names = list(markersAB.columns[1:]))
    return test_data

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask

        
class Imputer(sklearn.preprocessing.Imputer):
    """docstring for Imputer"""
    def __init__(self, weight=1., wc=None, *args, **kwargs):
        super(Imputer, self).__init__(*args, **kwargs)
        
        self.weight = weight
        self.wc = wc
        
    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        # Copy just once


        statistics = self.statistics_
        # Delete the invalid rows/columns
        invalid_mask = np.array(np.zeros_like(statistics), dtype=bool)
        valid_mask = np.logical_not(invalid_mask)
        valid_statistics = statistics[valid_mask]
        valid_statistics_indexes = np.where(valid_mask)[0]
        missing = np.arange(X.shape[not self.axis])[invalid_mask]

        if self.axis == 0 and invalid_mask.any():
            if self.verbose:
                warnings.warn("Deleting features without "
                              "observed values: %s" % missing)
            X = X[:, valid_statistics_indexes]

        # Do actual imputation

        mask = _get_mask(X, self.missing_values)
        n_missing = np.sum(mask, axis=self.axis)
        n_present = np.sum(np.logical_not(mask), axis=self.axis)

        if self.wc is not None:
            values = np.repeat(valid_statistics*np.where(n_present > self.wc, 1., n_present/float(self.wc)), n_missing)
        else:
            values = np.repeat(valid_statistics*self.weight, n_missing)

        if self.axis == 0:
            coordinates = np.where(mask.transpose())[::-1]
        else:
            coordinates = mask

        X[coordinates] = values
        X[np.isnan(np.array(X, dtype=np.float))]  = 0
        
        return np.array(X, dtype=np.float)
    
    def impute(self, X, y):
        return np.vstack((self.fit(X[y==1,:]).transform(X[y==1,:]) , 
                          self.fit(X[y==0,:]).transform(X[y==0,:])))


## Core pipeline
class HyPred(object):
    """docstring for HyPred"""
    def __init__(self, train_data=None, test_data=None, C=1.0, penalty='l1', acc_cutoff=0.95, cv_folds=10, classifier=None):
        self.train_data = train_data
        self.test_data = test_data
        self.C = C
        self.acc_cutoff = acc_cutoff
        self.cv_folds = cv_folds
        self.classifier = classifier
        self.penalty = penalty
        
        if self.classifier is None:
            self.classifier = lambda : LogisticRegression(class_weight='balanced', penalty=self.penalty, C=self.C, fit_intercept=False)
    
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
                self.results_data[contig] = Result(train=self.train_data[contig],
                                                  cv=cv,
                                                  contig=contig,   
                                                  classifier = self.classifier())
                self.results_data[contig].classifier.fit(X=self.train_data[contig].X, y=self.train_data[contig].y)
                
        print("Successfully trained %i classifiers" % len(self.selected_contigs))#
        
        return len(self.selected_contigs)
        
    def predict(self, use_het=True):
        """Predict ancestral population of origin, using test_data and the trained
         classifiers in results data, for the contigs in selected_contigs
        
        Params:
        results_data: dictionary {contig : Result} where the Result object has a classifier member populated by a trained classifier """
        
        self.pred_matrix = []
        self.p0_matrix = []
        for contig in self.selected_contigs:
            self.results_data[contig].test = self.test_data[contig]
            
            if use_het is True:
                try:
                    self.test_data[contig].growHetTree()
                except:
                    print "Failed growing genotype tree for contig {0}".format(contig)
                    continue
                    
                self.results_data[contig].het_tree_p = np.array([self.results_data[contig].classifier.predict_proba(x)[:,0] 
                                                                      for x in self.test_data[contig].het_tree])
                self.results_data[contig].karyotype = []
                self.results_data[contig].p0 = []
                
                for sample_i in range(self.results_data[contig].test.n_samples):
                    n_gen = self.results_data[contig].test.het_tree[sample_i].shape[0]
                    het_tree = self.results_data[contig].test.het_tree[sample_i]
                    genotype_p = self.results_data[contig].het_tree_p[sample_i].reshape(n_gen)
                    if n_gen > 1:
                        p = [genotype_p[:n_gen/2],genotype_p[:n_gen/2-1:-1]]
                        karyotype_p = (1-np.min(p,axis=0))*np.max(p, axis=0)
                        self.results_data[contig].karyotype.append((het_tree[np.argmax(karyotype_p)], het_tree[-(np.argmax(karyotype_p)+1)]))
                        self.results_data[contig].p0.append([np.min(p,axis=0)[0] ,np.max(p, axis=0)[0]]) 

                    else:
                        self.results_data[contig].p0.append([genotype_p[0],genotype_p[0]])
                        
                        
                self.p0_matrix.append(np.array(self.results_data[contig].p0)[:,0])
                self.p0_matrix.append(np.array(self.results_data[contig].p0)[:,1])
                self.pred_matrix.append(['/'.join(x) for x in self.results_data[contig].pred])
                
                del self.test_data[contig].het_tree
            else:
                self.results_data[contig].p0 = np.array(self.results_data[contig].classifier.predict_proba(self.test_data[contig].X))[:,0]
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
        
        A = np.sum(np.logical_or(self.pred_matrix == "popA",self.pred_matrix == "popA/popA"), axis=0)
        B = np.sum(np.logical_or(self.pred_matrix == "popB", self.pred_matrix == "popB/popB"), axis=0)
        C = np.sum(self.pred_matrix == "popA/popB", axis=0)
        XB = np.sum(self.pred_matrix == "X/popB", axis=0)
        AX = np.sum(self.pred_matrix == "popA/X", axis=0)
        X = np.sum(np.logical_or(self.pred_matrix == "X",self.pred_matrix == "X/X"), axis=0)
        
        ratio = np.array(A, dtype=np.float)/B
        
        print("Sample  \tpopA\tpopB\tpopA/popB\t\tno pred\tratio")
        print("-"*60)
        for i in range(len(sample_names)):
            print("{0}\t{1}\t{2}\t{3}\t{4}\t{5:.2f}".format(sample_names[i], A[i], B[i], C[i], AX[i]+XB[i]+X[i], ratio[i]))
        return (sample_names, A,B,C,XB,AX,X)
    def examine_contig(self, contig=None):
        if contig is None:
            contig = next(iter(self.results_data.keys()))
        
        plt.matshow(np.array(self.train_data[contig].X, dtype=np.float), vmax=1, vmin=-1, cmap=cm.seismic)
        plt.yticks(range(len(self.train_data[contig].sample_names)), self.train_data[contig].sample_names, size=4)
        plt.xticks(range(len(self.train_data[contig].pos)), self.train_data[contig].pos, rotation=90, size=8)
        plt.title("Training data")
        
        
        if self.test_data[contig].het_X is not None:

            
            arr = np.reshape([mp(x) for x in self.test_data[contig].het_X.flat], self.test_data[contig].het_X.shape)
            plt.matshow(np.array(arr, dtype=np.float), interpolation='none',cmap=col)
        
        else:
            plt.matshow(np.array(self.test_data[contig].X, dtype=np.float), vmax=1, vmin=-1, cmap=cm.seismic)
        plt.yticks(range(len(self.test_data[contig].sample_names)), self.test_data[contig].sample_names, size=4)
        plt.xticks(range(len(self.test_data[contig].pos)), self.test_data[contig].pos, rotation=90, size=8)
        plt.title("Test data")    
        
        print("Contig {0}\n".format(contig))
        
        print("\nCV Accuracy: {0:.2f}".format(np.mean(self.results_data[contig].cv)))
        
        plt.matshow(np.array(self.results_data[contig].classifier.coef_, dtype=np.float), 
                            vmax=abs(self.results_data[contig].classifier.coef_).max(), 
                            vmin=-abs(self.results_data[contig].classifier.coef_).max(), 
                            cmap=cm.seismic)
        plt.xticks(range(len(self.test_data[contig].pos)), self.test_data[contig].pos, rotation=90, size=8)
        plt.title("Weight matrix")
    
        
        
        print("Sample  \tPrediction\t Pr(B)")
        print("-"*40)
        for s,p, p0 in zip(self.test_data[contig].sample_names, self.results_data[contig].pred, self.results_data[contig].p0  ):
            if type(p0) is list:
                print("{0}\t{1}\t{2}".format(s,p,p0))
            else:
                print("{0}\t{1}\t{2:.2f}".format(s,p,p0))
            

