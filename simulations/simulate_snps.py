#!/usr/bin/env python

import pandas as pd
import numpy as np
import copy, os

num_to_lett = {1:'A',2:'T', 3:'G', 4:"C", -9:"X"}

@np.vectorize              
def mut(a,b):
    "mutate base a by amount b"
    if a == -9:
        return a
    else:
        return (a+b)%4 + 1
        
        
        
class Individual(object):
    """docstring for Individual"""

    def __init__(self, reference, genomeA=None, genomeB=None, ploidy='N', lineage=None, rng=None, recombMapA=None, recombMapB=None):
        self.reference = reference
        self.ploidy = ploidy
        self.genomeA = genomeA
        self.genomeB = genomeB
        self.lineage = lineage
        self.rng = rng
        self.recombMapA=recombMapA
        self.recombMapB=recombMapB
        
        if self.rng is None:
            self.rng = np.random.RandomState()
            
        if self.genomeA is None:
            self.genomeA = copy.deepcopy(self.reference)
        
        # Default polyploid indvidual to have to identical genomes
        if self.ploidy == 'N+N' or self.ploidy == '2N':
            if self.genomeB is None:
                self.genomeB = copy.deepcopy(self.genomeA)
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        
        ## Deep copy everything...
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
            
        ## Except the reference and the rng (very important)
        setattr(result, 'reference', self.reference)
        setattr(result, 'rng', self.rng)
        
        return result
        
    def mutate(self, mutation_mean, mutation_sd):
        """docstring for _mutateIndiv"""
        if self.ploidy == 'N':
            n_mut = np.array(np.abs(self.rng.normal(mutation_mean, mutation_sd)), dtype=np.int)
            mutation = np.array(self.rng.randint(0,3,n_mut), dtype=np.int)
            position = np.array(self.rng.randint(0,len(self.genomeA)-1, n_mut), dtype=np.int)
            self.genomeA[position] = mut(self.genomeA[position], mutation)
            
        else:
            n_mut = np.array(np.abs(self.rng.normal(mutation_mean, mutation_sd, 2)), dtype=np.int)
            
            mutationA = np.array(self.rng.randint(0,3,n_mut[0]), dtype=np.int)
            mutationB = np.array(self.rng.randint(0,3,n_mut[1]), dtype=np.int)
            
            positionA = np.array(self.rng.randint(0,len(self.genomeA)-1, n_mut[0]), dtype=np.int)
            positionB = np.array(self.rng.randint(0,len(self.genomeB)-1, n_mut[1]), dtype=np.int)
            
            self.genomeA[positionA] = mut(self.genomeA[positionA], mutationA)
            self.genomeB[positionB] = mut(self.genomeB[positionB], mutationB)

class Population(object):
    """docstring for Population"""
    def __init__(self, individuals, name, rng=None, reference=None):
        self.individuals = individuals
        self.name = name
        self.rng = rng
        self.reference = reference
        
        if self.rng is None:
            self.rng = np.random.RandomState()
    @property
    def n_contigs(self):
        return self.reference['contig'][-1]+1
           
    def mutate(self, mutation_mean, mutation_sd):
        """docstring for mutate"""
        for indv in self.individuals:
            indv.mutate(mutation_mean, mutation_sd)
            
    def write(self):
        """docstring for write"""
        contig_loci = ["{0}_{1}".format(c,l) for c,l in zip(self.reference['contig'], self.reference['loci'])]
        
        if self.individuals[0].ploidy == 'N':
            pop = np.vstack([indv.genomeA for indv in self.individuals], columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])])
                        
            for i in range(pop.shape[0]):
                for j in range(pop.shape[1]):
                    pop[i,j] = str(pop[i,j])*2
                    
            if self.individuals[0].recombMapA is not None:
                index = np.concatenate( ( np.atleast_2d([self.name]*self.n_contigs).T,
                                                np.atleast_2d(range(self.n_contigs)).T), axis=1)
                recombMapA = np.vstack([indv.recombMapA for indv in self.individuals])
                pd.DataFrame(np.concatenate( (index, np.atleast_2d(recombMapA).T ), axis=1 ),
                             columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])]).to_csv(self.name+".recombmapA", index=False)
            
                    
        else:
            genA = np.vstack([indv.genomeA for indv in self.individuals])
            genB = np.vstack([indv.genomeB for indv in self.individuals])
            
            pop = np.empty_like(genA)
        
            for i in range(pop.shape[0]):
                for j in range(pop.shape[1]):
                    pop[i,j] = str(genA[i,j])+str(genB[i,j])
                    
            if self.individuals[0].recombMapA is not None:
                index = np.concatenate( ( np.atleast_2d([self.name]*self.n_contigs).T,
                                                np.atleast_2d(range(self.n_contigs)).T), axis=1)
    
                recombMapA = np.vstack([indv.recombMapA for indv in self.individuals])
                pd.DataFrame(np.concatenate( (index, np.atleast_2d(recombMapA).T ), axis=1 ),
                             columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])]).to_csv(self.name+".recombmapA", index=False)
                
                recombMapB = np.vstack([indv.recombMapB for indv in self.individuals])
                pd.DataFrame(np.concatenate( (index, np.atleast_2d(recombMapB).T ), axis=1 ), 
                            columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])]).to_csv(self.name+".recombmapB", index=False)
                                
        
        df = pd.DataFrame( np.vstack( ([self.name]*len(contig_loci), contig_loci, pop)).T, columns = ['group', 'markers']+['sample'+str(i) for i in range(pop.shape[0])])
        df.to_csv(self.name+'.csv', index=False)
    
    def stats(self):
        """docstring for stats"""
        print("\nPopulation {0} statistics\n".format(self.name) + "-"*60 )
        het = [np.sum(x.genomeA != x.genomeB)/float(len(x.genomeA)) for x in self.individuals]
        print("Heterokaryotic ratio {0}".format("\t".join(["{0:.3f}".format(x) for x in  het])))
        print("Mean heterokaryotic ratio {0}".format(np.mean(het)))
        print("-"*60)

    @property
    def n_individuals(self):
        return len(self.individuals)

class Simulation(object):
    """docstring for Simulation"""
    def __init__(self, reference=None, random_seed=42, ploidy='N'):
        self.reference = reference
        self.rng = np.random.RandomState(random_seed)
        self.populations = {}
        self.ploidy = ploidy
    
    def create_reference(self, n_contigs, markers_mean, prefix=""):
        
        self.n_contigs = n_contigs
        self.markers_per_contig = np.array(self.rng.exponential(markers_mean, size=n_contigs), dtype=np.int) + 1
        self.reference = {}
        
        self.reference['loci'] = np.zeros(np.sum(self.markers_per_contig), dtype=np.int)
        self.reference['base'] = np.zeros_like(self.reference['loci'], dtype=np.int)
        self.reference['contig'] = np.zeros_like(self.reference['loci'], dtype=np.int)
        
        idx=0
        for contig in range(n_contigs):
            old_idx = idx
            idx += self.markers_per_contig[contig]
            self.reference['loci'][old_idx:idx] = np.arange(self.markers_per_contig[contig])
            self.reference['base'][old_idx:idx] = np.random.randint(1,5,self.markers_per_contig[contig])
            self.reference['contig'][old_idx:idx] = contig
        
        self.populations['reference'] = Population([Individual(genomeA=copy.deepcopy(self.reference['base']),
                                                               reference=self.reference,
                                                               ploidy=self.ploidy,
                                                               rng=self.rng)], name='reference', rng=self.rng, reference=self.reference)
    
    def divide(self, in_pop, sizes, out_pops=None, replace=True):
        """Either divides a population of individuals in_pop into new populations 
          out_pops each with the number individuals in the corresponding sizes element 
        or if no out_pop is specified divides in_pop into sizes new individuals"""
        
        if out_pops is None:
            out_pops = [in_pop]
            sizes = [sizes]
            
        assert len(out_pops) == len(sizes), "All subpopulations must have sizes defined"
        if replace is False:
            assert self.populations[in_pop].n_individuals >= np.sum(sizes), "To sample without replacement need more individuals in the in_pop than in all the out_pops"
            
        in_pop = self.populations[in_pop]
        del self.populations[in_pop.name]
        
        for i,pop in enumerate(out_pops):
            choices = self.rng.choice(range(in_pop.n_individuals),
                                      size=sizes[i], 
                                      replace=replace)
            copied_indvs = [copy.deepcopy(in_pop.individuals[c]) for c in choices]
            self.populations[pop] = Population(copied_indvs, name=pop, rng=self.rng, reference=self.reference)
    
    def exchangeNuclei(self, popA, popB, popAB):
        """Create n_children individuals with genomeA coming from a random 
        individual in popA and genomeB from an individual in popB"""
        
        parentsA = self.rng.choice(self.populations[popA].individuals)
        parentsB = self.rng.choice(self.populations[popB].individuals)
        self.populations[popAB] = Population([], name=popAB, rng=self.rng, reference=self.reference)
        
        hybrid = copy.deepcopy(parentsA)

        hybrid.recombMapA  = [popA]*self.n_contigs
        hybrid.genomeB = copy.deepcopy(parentsB.genomeB)
        hybrid.recombMapB  = [popB]*self.n_contigs
        
        self.populations[popAB].individuals.append(hybrid)
    
    def recombineContigs(self, popA, popB, popAB, n_children, n_parentsA=None, n_parentsB=None, ratio=0.5):
        """docstring for recombine"""
        if n_parentsA is None:
            n_parentsA = len(self.populations[popA].individuals)
        if n_parentsB is None:
            n_parentsB = len(self.populations[popB].individuals)
            


        self.populations[popAB] = Population([], name=popAB, rng=self.rng, reference=self.reference)
        
        if self.ploidy == 'N':
            for i in range(n_children):
                parentsA = self.rng.choice(self.rng.choice(self.populations[popA].individuals, size=n_parentsA, replace=False), size=n_children)
                parentsB = self.rng.choice(self.rng.choice(self.populations[popB].individuals, size=n_parentsB, replace=False), size=n_children)
                
                genA,recombMapA,_,_ = self.recombineHaploid(parentsA[i].genomeA, parentsB[i].genomeA, ratio=ratio, popA=popA, popB=popB)
                self.populations[popAB].individuals.append(Individual(genomeA=np.copy(genA), 
                                                                      recombMapA=recombMapA,
                                                                      reference=self.reference,
                                                                      ploidy=self.ploidy,
                                                                      rng=self.rng))
        elif self.ploidy == 'N+N':
            ## Select the individuals involved in the hybridisations 
            individualsA = self.rng.choice(self.populations[popA].individuals, size=n_parentsA, replace=False)
            individualsB = self.rng.choice(self.populations[popB].individuals, size=n_parentsB, replace=False)
            
            ## For each child randomly select its parents from the hybridizing pool 
            parentsA = self.rng.choice(individualsA, size=n_children)
            parentsB = self.rng.choice(individualsB, size=n_children)
            
            ## For each individual pick which genome is going to hybridize
            genomesA = [indv.genomeA if self.rng.rand() > 0.5 else indv.genomeB for indv in parentsA]
            genomesB = [indv.genomeA if self.rng.rand() > 0.5 else indv.genomeB for indv in parentsB]

            for i in range(n_children):

                genA, recombMapA, genB, recombMapB = self.recombineHaploid(genomesA[i], genomesB[i], ratio=ratio, popA=popA, popB=popB) 
                self.populations[popAB].individuals.append(Individual(genomeA=np.copy(genA),
                                                                      genomeB=np.copy(genB), 
                                                                      recombMapA=recombMapA,
                                                                      recombMapB=recombMapB,
                                                                      reference=self.reference,
                                                                      ploidy=self.ploidy,
                                                                      rng=self.rng))
    
    def recombineHaploid(self, genA, genB, ratio, popA, popB):
        """docstring for recombineHaploid"""
        parent_mask = []
        parent = self.rng.uniform(size=len(self.markers_per_contig)) > ratio
        for contig,_ in enumerate(self.markers_per_contig):
            parent_mask += [parent[contig] for i in range(self.markers_per_contig[contig])]
        return np.where(parent_mask, genA, genB), np.where(parent, popA, popB), np.where(parent_mask, genB, genA), np.where(parent, popB, popA)
    
    def write(self, folder):
        """docstring for write"""
        print("Writing reference to {0}".format(os.path.join(folder, "reference.txt")))
        
        with open(os.path.join(folder, "reference.txt"), 'w') as f:
            f.write("marker\tref\n")
            for l,c, b in zip(self.reference['loci'], self.reference['contig'], self.reference['base']):
                f.write("{0}_{1}\t{2}\n".format(c, l, num_to_lett[b]))
        
        for pop in self.populations.values():
            print("Writing population {0}".format(pop.name))
            pop.write()

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


