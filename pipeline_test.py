#!/usr/bin/env python
from hybridisation_pipeline import (markers_1, markers_4, markers_41, ref, loci, # Data
                                    encode, contig_bunch, create_training_data, # Methods
                                    create_test_data, train_predict, plot, examine_contig) 
                                    
                                    
from snp_encodings import discard_het, half_count, label 

import numpy as np
import matplotlib.pyplot as plt 
"""Label Group 1 = 1
         Group 4 = 0"""


################## half_count ######################################################
marker_1_enc = encode(markers_1, ref, half_count)
marker_4_enc = encode(markers_4, ref, half_count)            
hybrid_enc = encode(markers_41, ref, half_count)            
            
marker_1_bunched = contig_bunch(marker_1_enc, loci)
marker_4_bunched = contig_bunch(marker_4_enc, loci)
hybrid_bunched = contig_bunch(hybrid_enc, loci)

unfiltered = create_training_data(marker_1_bunched, marker_4_bunched)
hc_test_data = create_test_data(hybrid_bunched)
hc_training_data = {k:v for k,v in unfiltered.items() if np.sum(np.abs(v['X'])) > 1000}

 
## Now select contigs that look reasonable,
## Filter for contig

snps_per_contig = [contig['X'].shape[1] for contig in unfiltered.values()]
score_per_contig = [np.sum(np.abs(contig['X'])) for contig in unfiltered.values()]


plt.hist(score_per_contig, bins=25)
plt.savefig("score_per_contig.pdf")
plt.close()

plt.hist(snps_per_contig, bins=25)
plt.savefig("snps_per_contig.pdf")
plt.close()



########################################################################


################## discard het ######################################################v



unfiltered = create_training_data(contig_bunch(encode(markers_1, ref, discard_het), loci), 
                                  contig_bunch(encode(markers_4, ref, discard_het), loci))

dh_training_data = {k:v for k,v in unfiltered.items() if np.sum(np.abs(v['X'])) > 1000}
dh_test_data = create_test_data(contig_bunch(encode(markers_41, ref, discard_het), loci))


snps_per_contig = [contig['X'].shape[1] for contig in unfiltered.values()]
score_per_contig = [np.sum(np.abs(contig['X'])) for contig in unfiltered.values()]

plt.hist(score_per_contig, bins=25)
plt.savefig("score_per_contig.pdf")
plt.close()

plt.hist(snps_per_contig, bins=25)
plt.savefig("snps_per_contig.pdf")
plt.close()




train_predict(1.0, dh_training_data, dh_test_data)


##########################################################################################


