#!/usr/bin/env python
from hybridisation_pipeline import load_data, create_training_data, create_test_data, HyPred  
from snp_encodings import discard_het


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib
"""Label Group A = 1
         Group B = 0"""

def compare_to_truth(hp):
    
        
    print("{} contigs tested".format( len(list(hp.results_data.keys())) *  len(next(iter(hp.test_data.values())).sample_names) ))
    
    print ("Contigs not assigned {0}".format( np.sum([ np.sum(x.pred == 'X') for x in hp.results_data.values()]) ))
    
    print ("Contigs correctly identified as popA {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popA'],
                                                       hp.results_data[contig].pred == ['popA']))) 
                                                       for contig in hp.results_data.keys()] )))
                                    
    print ("Contigs from popA misidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popA'],
                                                       hp.results_data[contig].pred == ['popB']))) 
                                                       for contig in hp.results_data.keys()] )))
    print ("Contigs from popA unidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popA'],
                                                       hp.results_data[contig].pred == ['X']))) 
                                                       for contig in hp.results_data.keys()] )))
                                                       
    
    print ("Contigs correctly identified as popB {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popB'],
                                                       hp.results_data[contig].pred == ['popB']))) 
                                                       for contig in hp.results_data.keys()] )))
                                    
    print ("Contigs from popB misidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popB'],
                                                       hp.results_data[contig].pred == ['popA']))) 
                                                       for contig in hp.results_data.keys()] )))
    print ("Contigs from popB unidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == ['popB'],
                                                       hp.results_data[contig].pred == ['X']))) 
                                                       for contig in hp.results_data.keys()] )))



def load_truth(hp, recombmap_file):
    truth_df = pd.read_csv(recombmap_file, index_col=0)
    for contig in hp.test_data.keys():
        hp.test_data[contig].truth = np.array(truth_df.ix[int(contig)])
        


markersA, markersB, markersAB, ref, loci = load_data("simulations/popA.csv", "simulations/popB.csv", "simulations/popAB.csv", "simulations/reference.txt")
training_data = create_training_data(markersA, markersB, discard_het, loci, ref)
test_data  = create_test_data(markersAB, discard_het, loci, ref) 
hp = HyPred(train_data=training_data, test_data=test_data, C=1.)
hp.train()
hp.predict()