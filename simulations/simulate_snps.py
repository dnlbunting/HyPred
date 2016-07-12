#!/usr/bin/env python

import pandas as pd
import numpy as np
num_to_lett = {1:'A',2:'T', 3:'G', 4:"C", -9:"X"}

def create_referece(n_contigs, markers_mean, prefix=""):
    loci, base, contig_mask = [], [], []
    
    markers_per_contig = np.array(np.floor(np.random.exponential(markers_mean, size=n_contigs)), dtype=np.int)
    
    for contig in range(n_contigs):
          loci += [prefix+str(contig)+'_'+str(i) for i in range(markers_per_contig[contig])]
          base += list(np.random.randint(1,5,markers_per_contig[contig]))
          contig_mask +=[contig]*markers_per_contig[contig]
          
    return loci, base, markers_per_contig
def mut(a,b):
    "mutate base a by amount b"
    if a == -9:
        return a
    else:
        return (a+b)%4 + 1
def diverge(reference, mutation_mean, mutation_sd, n_children):
    
    children = [np.copy(reference) for i in range(n_children)]
    n_mut = np.array(np.floor(np.abs(np.random.normal(mutation_mean, mutation_sd, n_children))), dtype=np.int)
    
    for i in range(n_children):
        mutation = np.array(np.random.randint(0,3,n_mut[i]), dtype=np.int)
        position = np.array(np.random.randint(0,len(reference)-1, n_mut[i]), dtype=np.int)
        
        for j in range(n_mut[i]):
            children[i][position[j]] = mut(children[i][position[j]], mutation[j])
    return children    
def recombineIndiv(indA, indB, ratio, markers_per_contig):
    
    parent = np.random.uniform(size=len(markers_per_contig)) > ratio
    parent_mask = []
    for contig in range(len(markers_per_contig)):
        parent_mask += [parent[contig] for i in range(markers_per_contig[contig])]
        
    return np.where(parent_mask, indA, indB), np.where(parent,'popA', 'popB')
        
def recombine(popA, popB, ratio, markers_per_contig, n_children):
    parentsA = np.random.randint(0,len(popA)-1, n_children)
    parentsB = np.random.randint(0,len(popB)-1, n_children)
    res = [recombineIndiv(popA[a],popB[b], ratio, markers_per_contig) for a,b in zip(parentsA, parentsB)]
    return [x[0] for x in res], [x[1] for x in res]
    
def writeMarkers(pop, loci, name):
    pop = np.array(pop, dtype=np.object)
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            pop[i,j] = str(pop[i,j])*2
            
    df = pd.DataFrame( np.vstack( ([name]*len(loci), loci, pop)).T, columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])])
    df.to_csv(name+'.csv', index=False)
    
def writeReference(loci, base, name):
    with open(name, 'w') as f:
        f.write("marker\tref\n")
        for l,b in zip(loci, base):
            f.write(str(l)+"\t"+num_to_lett[b]+"\n")

def writeRecombMap(recombMap, name):
    with open(name+".recombmap",'w') as f:
        f.write('contig,'+ ','.join(['sample'+str(i) for i in range(len(recombMap))]) +'\n' )
        for i,m in enumerate(np.transpose(recombMap)):
            f.write("{0},{1}\n".format(i,','.join(m)))
        
            
                
def simulate_missing_data(pop, rate_mean, rate_sd):
    
    n_errs = np.array(np.floor(len(pop[0])*np.abs(np.random.normal(rate_mean, rate_sd, len(pop)))), dtype=np.int)
    for i in range(len(pop)):
         position = np.array(np.random.randint(0,len(pop[i])-1, n_errs[i]), dtype=np.int)
         for j in range(n_errs[i]):
             pop[i][position[j]] = -9
             
    return pop

    


## From the dataset:
## n_contigs =5000
## mean_markers_per_contig = 7.5
## mean 

def simulate_hybrid():
    
    loci, base, markers_per_contig = create_referece(5000,7.5)
    a, b = diverge(base, 18000, 2500, 2)
    
    ## Simulate lineage specific deletions 
    a, = simulate_missing_data([a], 0.1,0.05)
    b, = simulate_missing_data([b], 0.1,0.05)
    
    
    popA = diverge(a, 1000, 250, 25)
    popB = diverge(b, 1000, 250, 25)
    
    popC, recombination_map = recombine(popA, popB, 0.1, markers_per_contig, 7)
    popC = diverge(c, 1000, 250, 10)
    
    popA = simulate_missing_data(popA, 0.5,0.05)
    popB = simulate_missing_data(popB, 0.5,0.05)
    popC = simulate_missing_data(popC, 0.5,0.05)
    
    writeRecombMap(recombination_map, 'popAB')
    writeMarkers(popA, loci, "popA")
    writeMarkers(popB, loci, "popB")
    writeMarkers(popC, loci, "popAB")
    writeReference(loci, base, "reference.txt")


def simulate_no_hybrid():
    loci, base, markers_per_contig = create_referece(5000,7.5)
    a, b = diverge(base, 18000, 2500, 2)
    b,c = diverge(b, 10000,2500, 2)
    
    popA = diverge(a, 1000, 250, 34)
    popB = diverge(b, 1000, 250, 100)
    popC = diverge(c, 1000, 250, 10)
    
    popA = simulate_missing_data(popA, 0.5,0.05)
    popB = simulate_missing_data(popB, 0.5,0.05)
    popC = simulate_missing_data(popC, 0.5,0.05)
    
    
    writeMarkers(popA, loci, "popA")
    writeMarkers(popB, loci, "popB")
    writeMarkers(popC, loci, "popAB")
    writeReference(loci, base, "reference.txt")


