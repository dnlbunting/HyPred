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

def load_data(markersA_file, markersB_file, markersAB_file, ref_file):
    
    markersA = pd.read_csv(markersA_file, comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    markersB = pd.read_csv(markersB_file, comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    markersAB = pd.read_csv(markersAB_file, comment='#', skip_blank_lines=True).drop("group", 1).dropna(axis=0)
    
    reference = pd.read_csv(ref_file, comment='#', sep='\t', skip_blank_lines=True)
    ref = { x[0]:str(lett_to_num[x[1]]) for x in reference.to_dict(orient='split')['data']}
    
    assert np.all(markersA['markers'] == markersB['markers'])
    assert np.all(markersA['markers'] == markersAB['markers'])
    
    loci = markersA['markers']
    
    return markersA, markersB, markersAB, ref, loci



def encode(marker_df, ref, enc_function):
    """ Encodes the dataframe of bases at each locus per library under the encoding scheme given in enc_function
    
        Params:
        marker_df : dataframe of SNPs for each library from Vanessa
        ref: dictionary mapping contig_pos -> {1,2,3,4} = {A.T, G, C}
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

def create_training_data(bunched1, bunched0):
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
    
    contig_X = {}
    
    for c in bunched0.keys():
        assert bunched1[c]['pos'] == bunched0[c]['pos']
        contig_X[c] = {}
        contig_X[c]['pos'] =  bunched0[c]['pos']
        contig_X[c]['X'] = np.concatenate(( bunched1[c]['data'], bunched0[c]['data']), axis=1)
        contig_X[c]['X'] = np.transpose(contig_X[c]['X'])
        contig_X[c]['y'] = np.array([1]*bunched1[c]['data'].shape[1] + [0]*bunched0[c]['data'].shape[1])
        
    return contig_X

def create_test_data(bunched):
    
    """docstring for create_test_data"""
    test_data = {}
    for c in bunched.keys():
        test_data[c] = {}
        test_data[c]['pos'] = bunched[c]['pos']
        test_data[c]['X'] = np.transpose(bunched[c]['data'])
        
    return test_data

def train_predict(C, train_data, test_data, plot=False):
    selected_contigs = list(train_data.keys())
    
    classifier = lambda : LogisticRegression(class_weight='balanced', penalty='l1',C=C, fit_intercept=False)
    print("\n" + '-'*60 + "\n C = "+str(C)+"\n"+'-'*60)
    
    cv_accuracy = []
    cv_selected_contigs = []
    for contig in train_data.keys():    
        train_data[contig]['cv'] = cross_validation.cross_val_score(classifier(), X=train_data[contig]['X'], y=train_data[contig]['y'], 
                                         cv=cross_validation.StratifiedKFold(train_data[contig]['y'], 10, shuffle=True))
                                         
        acc = np.mean(train_data[contig]['cv'])
        cv_accuracy.append(acc)
        
        ## Require CV accuracy of >95% to use to evaluate hybridisation 
        if acc > 0.95:
            cv_selected_contigs.append(contig)    
            train_data[contig]['lr'] = classifier()
            train_data[contig]['lr'].fit(X=train_data[contig]['X'], y=train_data[contig]['y'])
    
        
    print("Successfully trained %i classifiers" % len(cv_selected_contigs))#
    
    
    results = []    
    results_contigs = {}
    for contig in cv_selected_contigs:
        test_data[contig]['p0'] = [x[0] for x in train_data[contig]['lr'].predict_proba(test_data[contig]['X'])]
        results_contigs[contig] = test_data[contig]
        results.append(test_data[contig]['p0'])
        results_contigs[contig]['lr'] = train_data[contig]['lr']
    
    results = np.array(results)
    
    print(">0.9 : " + ' '.join([str(x) for x in np.sum(results > 0.9, axis=0)]))
    print("<0.1 : " + ' '.join([str(x) for x in np.sum(results < 0.1, axis=0)]))
    
    
    if plot==True:
        plt.figure()
        plt.hist(cv_accuracy, bins=50)
        plt.figure()
        for i in range(results.shape[1]):
            plt.hist(results[:,i],bins=50, histtype='step')
            plt.xlabel("Pr(Group 4)")
            
    return results_contigs

def plot(pssm, cb=True):
    '''
    Plots the S matrix
    '''     
    pssm = np.array(pssm, dtype=np.float)
    plt.matshow(pssm, cmap = cm.seismic, vmax=abs(pssm).max(), vmin=-abs(pssm).max())
    if cb:
        plt.colorbar()
    plt.show()


def examine_contig(contig, train, result):
    plot(result[contig]['X'], cb=False)
    plt.title("Group 41")
    
    plot(train[contig]['X'], cb=False)
    plt.title("Group 1 and Group 4")
    
    plot(result[contig]['lr'].coef_, cb=False)
    plt.title("Classifier weights")
    