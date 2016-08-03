#!/usr/bin/env python
import numpy as np

def _encode_het(obs, ref):
    if '-9' in obs or obs == float('nan'):
        ## Missing data
        return [0]
        
    elif obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return [-1]
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return [-1,+1]
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return [+1]

    else:
        ## Heterokaryotic, neither match reference
        return [+1]
        
encode_het = np.vectorize(_encode_het, otypes=[np.object])

def _discard_het(obs, ref, stats=None):
    if '-9' in obs or obs == float('nan'):
        ## Missing data
        stats['missing']+=1
        return 0
        
    elif obs == ref*2:
        ## Homokaryotic match for refetrecne 
        stats['Homokaryotic reference']+=1
        return -1
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        stats['Heterokaryotic reference/variant']+=1
        return 0
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        stats['Homokaryotic variant']+=1
        return 1

    else:
        ## Heterokaryotic, neither match reference
        stats['Heterokaryotic variant']+=1
        return 0
discard_het = np.vectorize(_discard_het, excluded = ('stats'),otypes=[np.float])

def enc2bit(obs_arr, ref_arr, stats=None):
    encoded = np.empty((obs_arr.shape[0]*2, obs_arr.shape[1]))
    print obs_arr.shape
    for loci_idx, ref in enumerate(ref_arr.flat):
        for sample_idx, obs in enumerate(obs_arr[loci_idx,:].flat):
            
            if '-9' in obs or obs == float('nan'):
                ## Missing data
                encoded[2*loci_idx, sample_idx] = 0
                encoded[2*loci_idx+1, sample_idx] = 0
                stats['missing']+=1
                
            elif obs == ref*2:
                ## Homokaryotic match for refetrecne 
                encoded[2*loci_idx, sample_idx] = -1
                encoded[2*loci_idx+1, sample_idx] = -1
                stats['Homokaryotic reference']+=1
                
            elif ref in obs:
                ## Heterokaryotic, one is match for reference
                encoded[2*loci_idx, sample_idx] = -1
                encoded[2*loci_idx+1, sample_idx] = +1
                stats['Heterokaryotic reference/variant']+=1
                
            elif obs[0] == obs[1]:
                ## Homokaryotic, different from reference
                encoded[2*loci_idx, sample_idx] = +1
                encoded[2*loci_idx+1, sample_idx] = +1
                stats['Homokaryotic variant']+=1
                
            else:
                ## Heterokaryotic, neither match reference
                encoded[2*loci_idx, sample_idx] = +1
                encoded[2*loci_idx+1, sample_idx] = +1
                stats['Heterokaryotic variant']+=1
                
    return encoded
                
            
def half_count(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return -1
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return 0
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return +0.5
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return 1

    else:
        ## Heterokaryotic, neither match reference
        return 1

def label(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return 'rr'
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return 'e'
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return 'rx'
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return 'xx'

    else:
        ## Heterokaryotic, neither match reference
        return 'xy'
        
def null_encoding(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return 1
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return 0
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return 1
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return 1

    else:
        ## Heterokaryotic, neither match reference
        return 1
        
def sparse_encoding(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return np.array([1,0,0])
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return np.array([0,0,1])
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return np.array([0,1,0])
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return np.array([-1,0,0])

    else:
        ## Heterokaryotic, neither match reference
        return np.array([-1,0,0,0])        
