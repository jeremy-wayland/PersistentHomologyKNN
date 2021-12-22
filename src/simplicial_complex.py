import numpy as np
import pandas as pd
import networkx as nx
import warnings
from itertools import combinations


class SimplicialComplex:
    """
    This class generates a simplicial complex for a given set of data
    and a given radius to determine adjacency between points.
    After initialization, points may be added or removed from the complex.
    The class can return, for a given point, what simplices the point belongs to
    within the complex, and the distance, area, or generalized volume between points
    represented by that simplex. The point need not already be a member of the complex.
    The class can return, for a given point, the nearest k neighbors where:
            All points are considered vertices on a weighted undirected graph
                determined by the 1-simplices of the simplicial complex. The weights
                will correspond to the distance between the connected vertices.
            The neighbors of a point p are all points reachable from p by transversing
                the edges of the graph.
            Distance between two points (p,q) is determined by the shortest path along
                the edges of the graph that connects p and q. The distance will be
                the sum of the weights along this path. If no path exists between
                p and q, then the distance between them will be undefined.
    """
    def __init__(self,
                radius,
                graph_distance_metric=None,
                weight_distance_metric=None,
                vertices=None,
                delim=','):
        ####CLASS INSTANCE VARIABLES
        self.radius = 2*radius
        self.graph_distance_metric = graph_distance_metric
        self.weight_distance_metric = weight_distance_metric
        self.complex = nx.Graph()
        self.node_shape = None

        #define euclidean distance
        def euclidean(x,y):
            return np.linalg.norm(x-y)
        #If either distance metric is set to None, default the metric to
        #euclidean distance
        if self.graph_distance_metric is None:
            self.graph_distance_metric = euclidean
        if self.weight_distance_metric is None:
            self.weight_distance_metric = euclidean
        #build simplicial complex if vertices are provided
        if vertices is not None:
            self.build_complex(vertices, delim)

    def build_complex(self, vertices, delim=None):
        #import set of points from file, np array or pandas dataframe
        points = self.import_vertices(vertices, delim)
        #add points
        for p in points:
            self.add_point(p)

    def add_point(self, point):
        point = np.array(point)
        assert self.node_shape is None or self.node_shape == point.shape,\
            f'New point must be of the same shape as data already in simplex. '\
            f'Expected {self.node_shape}, got {point.shape}'
        #set shape of simplex data if this is the first point added to simplex
        if self.node_shape is None:
            self.node_shape = point.shape
        # convert point to hashable tuple
        point_id = tuple(point)
        # do nothing if point already exists in complex
        if self.complex.has_node(point_id):
            return
        self.complex.add_node(point_id)
        distance_vector = self.get_distance_vector(point)
        for graph_distance, weight_distance, node in distance_vector:
            if graph_distance <= self.radius and graph_distance != 0:
                # convert neighbor to hashable tuple
                neighbor_id = tuple(node)
                self.complex.add_edge(point_id,
                                      neighbor_id,
                                      weight=weight_distance)

    def remove_point(self, point):
        point_id = tuple(point)
        if self.complex.has_node(point_id):
            self.complex.remove_node(point_id)

    def has_point(self, point):
        point_id = tuple(point)
        return self.complex.has_node(point_id)

    def get_distance_vector(self, point):
        '''
        Returns list of tuples (graph_distance, weight_distance, node) of
        the graph distance and weight distance between the point parameter
        and all nodes (corresponding to the third tuple element) in the complex.
        '''
        graph_distance = \
        [self.graph_distance_metric(point,np.array(x)) for x in self.complex.nodes]
        if self.graph_distance_metric == self.weight_distance_metric:
            weight_distance = graph_distance
        else:
            weight_distance = \
                [self.weight_distance_metric(point,np.array(x)) for x in self.complex.nodes]
        distances = \
            zip(graph_distance, weight_distance, list(self.complex.nodes))
        return list(distances)

    def import_vertices(self, vertices, delim=','):
        assert type(vertices) in [str, list, np.ndarray, pd.DataFrame],\
            'Vertices input of incorrect type, must be list of numeric tuples,\
            filepath string, Pandas Dataframe, or Numpy ndarray'
        if type(vertices)==str:
            with open(vertices) as f:
                return np.loadtxt(f, delimiter=delim)
        else:
            return np.array(vertices)