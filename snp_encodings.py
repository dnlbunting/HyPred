#!/usr/bin/env python

def encode_het(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return [-1]
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return [0]
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return [-1,+1]
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return [+1]

    else:
        ## Heterokaryotic, neither match reference
        return [+1]
    
def discard_het(ref, obs):
    
    if obs == ref*2:
        ## Homokaryotic match for refetrecne 
        return -1
        
    elif obs == '-9-9' or obs == float('nan'):
        ## Missing data
        return 0
        
    elif ref in obs:
        ## Heterokaryotic, one is match for reference
        return 0
        
    elif obs[0] == obs[1]:
        ## Homokaryotic, differnt from reference
        return 1

    else:
        ## Heterokaryotic, neither match reference
        return 0

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
