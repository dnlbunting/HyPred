#!/usr/bin/env python
from hybridisation_pipeline import (encode, contig_bunch, create_training_data, create_test_data, train_predict, plot, examine_contig, load_data)
from snp_encodings import discard_het, half_count, label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib
"""Label Group A = 1
         Group B = 0"""

def compare_to_truth(results, recombmap_file):
    
    truth_df = pd.read_csv(recombmap_file, index_col=0)
    
    for contig in results.keys():
        results[contig]['pred'] = []
        for lib in range(len(results[contig]['p0'])):
            if results[contig]['p0'][lib] < 0.1 :
                results[contig]['pred'].append('popA')
            elif results[contig]['p0'][lib] > 0.9 :
                results[contig]['pred'].append('popB')
            else:
                results[contig]['pred'].append('X')
        results[contig]['pred'] = np.array(results[contig]['pred'])
        
        
    print("{} contigs tested".format(len(list(results.keys()))*7))
    
    print ("Contigs not assigned {0}".format(np.sum([np.sum(x['pred'] == np.array('X')) for x in results.values()])))
    
    print ("Contigs correctly identified as popA {0}".format((
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popA']),
                                                       results[contig]['pred'] == np.array(['popA'])))) 
                                                      for contig in results.keys()]))))
                                    
    print ("Contigs from popA misidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popA']),
                                                       results[contig]['pred'] == np.array(['popB'])))) 
                                                       for contig in results.keys()])))
    print ("Contigs from popA unidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popA']),
                                                       results[contig]['pred'] == np.array(['X'])))) 
                                                       for contig in results.keys()])))
                                                       
    
    print ("Contigs correctly identified as popB {0}".format((
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popB']),
                                                       results[contig]['pred'] == np.array(['popB'])))) 
                                                      for contig in results.keys()]))))
                                    
    print ("Contigs from popB misidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popB']),
                                                       results[contig]['pred'] == np.array(['popA'])))) 
                                                       for contig in results.keys()])))
    print ("Contigs from popB unidentified {0}".format(
                        np.sum([(np.sum(np.logical_and(np.array(truth_df.ix[int(contig)]) == np.array(['popB']),
                                                       results[contig]['pred'] == np.array(['X'])))) 
                                                       for contig in results.keys()])))


markersA, markersB, markersAB, ref, loci = load_data("simulations/popA.csv", "simulations/popB.csv", "simulations/popAB.csv", "simulations/reference.txt")
unfiltered = create_training_data(contig_bunch(encode(markersA, ref, discard_het), loci),
                                  contig_bunch(encode(markersB, ref, discard_het), loci))

snps_per_contig = [contig['X'].shape[1] for contig in unfiltered.values()]
score_per_contig = [np.sum(np.abs(contig['X'])) for contig in unfiltered.values()]

plt.hist(score_per_contig, bins=25)
plt.hist(snps_per_contig, bins=25)



training_data = {k:v for k,v in unfiltered.items() if np.sum(np.abs(v['X'])) > 1000}
test_data = create_test_data(contig_bunch(encode(markersAB, ref, discard_het), loci))
results = train_predict(1.0, training_data, test_data,plot=True)
compare_to_truth(results, "simulations/popAB.recombmap")

