# Using Ripser.py to extract radii that can generate Vietoris Rips Complexes with desirable topological features
# Author: Jeremy Wayland

import numpy as np
from ripser import ripser
from persim import plot_diagrams



def feature_selection(data,max_dim,min_persistence,delim=','):
    """This function will compute the persistence barcode of a point cloud using Ripser, and plot the diagram.
    Uses scikit-tda's ripser.py: a lean persistent homology package for Python.
    It will then filter for features that persist longer than a desired threshold.

    Inputs:
        data: txt file or numpy array
        max_dim: specifies the maximum dimension of homology to be computed
        min_persistence: threshold value for how long you want ~important~ features to persist for

    Returns:
        features: a dictonary of lifetimes of features seperated by by dimension
     
    """

    if type(data)==str:
        with open(data) as f:
            data = np.loadtxt(f, delimiter=delim)

    #Generate Persistence Diagram for Point Cloud and Visualize
    dgms = ripser(data,maxdim=max_dim)['dgms']
    plot_diagrams(dgms, show=True)

    
    #Convert to array
    barcodes = np.array(dgms,dtype=object)
    features = dict()

    for dim in range(max_dim,0,-1):
        barcode = barcodes[dim]
        persistent_features = []

        if len(features)>0:
            for interval in barcode:
                persistence = abs(interval[1] - interval[0]) #Death of feature - Birth of Feature
                #Save features that persist longer than min_persistence
                if persistence > min_persistence:
                    persistent_features.append(interval)
        
        #Ensure intervals are sorted by their lower bounds
        persistent_features.sort(key=lambda x:x[0])
        features[dim] = persistent_features
        
    return features
            
def interval_intersection(lst:list,flag=False):
    """A recursive function for sorting a list of intervals and evaluating the self intersetion $\bigCap$.
    Inputs:
        lst: a list of nested tuples (persistence intervals)
        flag: a paramater defaulting to False that tracks whether any intersection has occurred

    Returns:
        An interval equal to the self intersection of the original list.
        If no intersection occurs, this will return the interval with the largest lowerbound.

    """
    if len(lst) == 1:
        return lst[0],flag
    else:
        #Sort by lower bound
        lst.sort(key=lambda x:x[0])
        
        A = lst[0]
        B = lst[1]
       
        #Check for intersection?
        if min(B) <= max(A):
            lst[1] = np.array([min(B),min(max(A),max(B))]) 
            flag=True
        else:
            if flag:
                #If there has been previous intersection then have this interval propapgate
                lst[1] = A
        #Recursive Call
        return interval_intersection(lst[1:],flag)
        

def candidate_intervals(filtered_pd:dict):
    """ This function recursively evaluates intervals that are candidates for preserving desired topological features that are identified by a persistence diagram.
    
    Inputs:
        filtered_pd: a filtered persistence diagram, encoded as a dictionay that seperates persistence lifetime by dimension.
            - designed to take a dictionary outputted by the feature_selection function.
    
    Returns
        list of candidate intervals that retain homology structure when converted to a VR complex. 
    """
    #Just for dim 2 since we are only concerned with cycles
    dim = max(filtered_pd.keys())
    #Get Max Nontrivial Dimension Recursively
    if len(filtered_pd[dim]) ==0 : #Trivial Barcode
        filtered_pd.pop(dim)
        if len(filtered_pd) == 0:
            print('No candidate intervals found. Try increasing minimum persistence threshold')
            return None
        return candidate_intervals(filtered_pd)
  
    cap,flag = interval_intersection(filtered_pd[dim])
    #Have dimensions collapsed?
    if dim == 1:
        #Is there nontrivial intersection
        if flag:
            return cap
            #returns list of all non-trivial homology intervals
            
        else:
            return filtered_pd[dim]

    #dim 2 or greater
    if flag:
        #Proceed element wise down to next dimension
            #Add all intervals to lower dimension
        for interval in filtered_pd[dim]:
            filtered_pd[dim-1].append(interval)
    else:
        filtered_pd[dim-1].append(cap)
    #Remove current dimension and proceed to next, recursively
    filtered_pd.pop(dim)

    return candidate_intervals(filtered_pd)



def select_radii(data,max_dim,min_persistence):
    """"" This function extracts radii that can generate Vietoris Rips Complexes with desirable topological features.

    Inputs:
        data: txt file or numpy array
        max_dim: specifies the maximum dimension of homology to be computed
        min_persistence: threshold value for how long you want ~important~ features to persist for

    Returns:
        a list of token radii.
    """
    features = feature_selection(data,max_dim,min_persistence)
    candidates = candidate_intervals(features)
    if candidates is None:
        return None
    
    return [np.mean(x) for x in candidates]
    #Take the median of each candidate interval
    