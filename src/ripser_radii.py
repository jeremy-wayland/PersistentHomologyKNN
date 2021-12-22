# A python wrapper for Ripser

# Automate ripser call 
# seperate bar codes by dimension
# Choose feature tuples above some persistence threshold
# Start with highest dimension and iteratively intersect with lower dimensions

import numpy as np
from ripser import ripser
from persim import plot_diagrams



def feature_selection(data,max_dim,min_persistence,delim=','):
    """This function will compute the persistence barcode of a point cloud using Ripser, and plot the diagram.

This function uses scikit-tda's ripser.py: a lean persistent homology package for Python.

Note:
    max_dim: specifies the maximum dimension of homology to be computed
    data: txt file or numpy array
    
     This will be done by selecting the betti numbers that persist longer than the threshold
        Then radii can be selected that preserve homology for each successive dimension. 
    """

    if type(data)==str:
        with open(data) as f:
            data = np.loadtxt(f, delimiter=delim)

    #Generate Persistence Diagram for Point Cloud and Visualize
    dgms = ripser(data,maxdim=max_dim)['dgms']
    plot_diagrams(dgms, show=True)

    
    #Convert to Numpy
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
            
def interval_intersection(array:list,flag=False):
    if len(array) == 1:
        return array[0],flag
    else:
        #Sort by lower bound
        array.sort(key=lambda x:x[0])
        
        A = array[0]
        B = array[1]
       
        #Check for intersection?
        if min(B) <= max(A):
            array[1] = np.array([min(B),min(max(A),max(B))]) 
            print(f'Intersection, new interval: {array[1]}')
            flag=True
        else:
            print('No intersection')
            if flag:
                #If there has been previous intersection then have this interval propapgate
                array[1] = A
        #Recursive Call
        return interval_intersection(array[1:],flag)
        

def candidate_intervals(filtered_pd:dict):
    #Just for dim 2 since we are only concerned with cycles
    dim = max(filtered_pd.keys())
    print(dim)
    #Get Max Nontrivial Dimension Recursively
    if len(filtered_pd[dim]) ==0 : #Trivial Barcode
        filtered_pd.pop(dim)
        print(f'Trimmed dictionary: {filtered_pd}')
        if len(filtered_pd) == 0:
            print('No candidate intervals found. Try increasing minimum persistence threshold')
            return None
        return candidate_intervals(filtered_pd)

    #make sure intervals are sorted  
    print(f'Im printing my persistence dictionary: {filtered_pd}')

    cap,flag = interval_intersection(filtered_pd[dim])
    print(f"The intersection is: {cap}")
    #Have dimensions collapsed?
    if dim == 1:
        print('Dimension is 1')
        #Is there nontrivial intersection
        print(filtered_pd[dim][-1])
        if flag:
            print('About to return, should not recurse farther')
            return cap
            #returns list of all non-trivial homology intervals
            
        else:
            return filtered_pd[dim]

    #dim 2 or greater
    if flag:
        print('No intersection confirmed, should procceed to next dimension')
        #Proceed element wise down to next dimension
            #Add all intervals to lower dimension
        for interval in filtered_pd[dim]:
            filtered_pd[dim-1].append(interval)
    else:
        filtered_pd[dim-1].append(cap)
    #Remove current dimension and proceed to next, recursively
    filtered_pd.pop(dim)
    print(f'Trimmed pd: {filtered_pd}')

    return candidate_intervals(filtered_pd)



def select_radii(data,max_dim,min_persistence):
    
    features = feature_selection(data,max_dim,min_persistence)
    candidates = candidate_intervals(features)
    
    #Take the median of each candidate interval
    return [np.mean(x) for x in candidates]