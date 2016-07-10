#!/usr/bin/env python

from hybridisation_pipeline import load_data, create_training_data, create_test_data, HyPred  
from snp_encodings import discard_het

import numpy as np
import matplotlib.pyplot as plt 


"""Label Group 1 = 1 = A
         Group 4 = 0 = B"""

markersA, markersB, markersAB, ref, loci = load_data("data/markers_1.txt", "data/markers_4.txt", "data/markers_41.txt", "data/reference.txt")
training_data = create_training_data(markersA, markersB, discard_het, loci, ref)
test_data  = create_test_data(markersAB, discard_het, loci, ref) 

hp = HyPred(train_data=training_data, test_data=test_data, C=0.75)
hp.train()
hp.predict()