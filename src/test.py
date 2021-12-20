from scipy.spatial.distance import pdist, squareform, cdist
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simplicial_complex import SimplicialComplex
from persistent_homology_knn import PHKnn
# Test construction of simplicial complex
print('Testing construtor using torus data')
sc = SimplicialComplex(.5, vertices='../Data/torus_points.txt')
#nx.draw(sc.complex)
#plt.show()
print('PASSED')

print('Testing contructor using tours np array')
with open('../Data/torus_points.txt') as f:
    torus_data = np.loadtxt(f, delimiter=',')
sc = SimplicialComplex(.5, vertices=torus_data)
#nx.draw(sc.complex)
#plt.show()
print('PASSED')

# Test point insertion
print('Testing regular point insertion')
sc.add_point(np.array((-1,1,1,1)))
print('PASSED')



print('Testing point insertion with different type')
sc.add_point([-1,1,1,1])
print('PASSED')

print('Testing point insertion with incorrect dimension')
try:
    sc.add_point(np.array((-1,1,-1)))
except AssertionError as err:
    print(err)
    print('PASSED')
# Test point removal
print('Testing point removal')
sc.remove_point(np.array((-1,1,1,1)))
if not sc.complex.has_node((-1,1,1,1)):
    print('PASSED')
else:
    print('FAILED')
    exit(0)

print('Testing removing point not in complex')
try:
    sc.remove_point(np.array((1,1,1,1,1)))
except AssertionError as err:
    print(err)
    print('PASSED')


# Testing PHKnn

#create training and testing data
with open('../Data/torus_points.txt') as f:
    X = np.loadtxt(f, delimiter=',')
y = []
for x in X:
    y.append(random.choice([1, 0]))
y = np.array(y)

print('Testing PHKnn constructor')
knn = PHKnn(.01,5)
print('PASSED')

print('Testing fit function')
knn.fit(X,y)
print('PASSED')

print('Testing predict function')
y = np.array([0,0,0,1])
print(knn.predict(y))
print('PASSED')


