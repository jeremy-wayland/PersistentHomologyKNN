import numpy as np
import pandas as pd
import networkx as nx
import warnings
from itertools import combinations


class Simplex:
    def __init__(self,
    points=None,
    graph=None,
    distance_metric=None,
    simplex=None):
        self.graph = graph
        self.distance_metric = distance_metric
        self.dimension = 0
        self.magnitude = 0
        self.perimeter = 0
        if simplex is not None:
            self.copy_simplex(simplex)
        if distance_metric is None:
            def euclidean(x, y):
                return np.linalg.norm(x-y)
            self.distance_metric = euclidean
        self.add_points(points)
        self.update_dimensions()

    def copy_simplex(simplex):
        self.graph = nx.copy(simplex.graph)
        self.distance_metric = simplex.distance_metric
        self.dimension = simplex.dimension
        self.magnitude = simplex.magnitude
        self.perimeter = simplex.perimeter

    def add_points(self, points):
        if points is not None:
            self.graph.add_nodes(points)
        self.fully_connect()

    def compose_simplices(self, simplex):
        assert type(simplex) in [simplicial_complex.Simplex, nx.Graph],\
        'simplex parameter must be another simplex or a complete weighted \
        undirected graph from Networkx'
        if type(simplex)==simplicial_complex.Simplex:
            graph = simplex.graph
        #ensure no edges are repeated in new graph
        for edge in graph.edges:
            if self.graph.has_edge(*edge):
                graph.remove_edge(*edge)
        #build composition of both simplicies
        new_graph = nx.compose(graph, self.graph)
        new_simplex = Simplex(graph=new_graph,
                              distance_metric=self.distance_metric)
        return new_simplex

    def absorb_simplex(self, simplex):
        new_simplex = self.compose_simplices(simplex)
        self.copy_simplex(new_simplex)

    def update_dimensions(self):
        edges = dict(self.graph.edges)
        magnitude=1
        perimiter=0
        for edge in edges:
            weight = edges[edge][weight]
            magnitude*=weight
            perimeter+=weight
        self.magnitude = magnitude
        self.perimeter = perimeter
        self.dimension = len(self.graph.nodes)

    def fully_connect(self):
        nodes = list(self.graph.nodes)
        edges = combinations(nodes,2)
        graph_updated = False
        for edge in edges:
            if not self.graph.has_edge(*edge):
                graph_updated = True
                weight = self.distance_metric(*edge)
                self.graph.add_edge(*edge)
        if graph_updated:
            update_dimensions()





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
        self.radius = radius
        self.graph_distance_metric = graph_distance_metric
        self.weight_distance_metric = weight_distance_metric
        self.complex = nx.Graph()

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
            complex = self.build_complex(vertices, delim)

    def build_complex(self, vertices, delim=None):
        #import set of points from file, np array or pandas dataframe
        points = self.import_vertices(vertices, delim)
        #add points
        for p in points:
            self.add_point(p)

    def add_point(self, point):
        assert isinstance(point, np.ndarray), 'Point parameter must be numpy array'
        # convert point to hashable tuple
        point_id = tuple(point)
        # do nothing if point already exists in complex
        if self.complex.has_node(point_id):
            return
        self.complex.add_node(point_id)
        distance_vector = self.get_distance_vector(point)
        for graph_distance, weight_distance, node in distance_vector:
            if graph_distance <= 2*self.radius and graph_distance != 0:
                # convert neighbor to hashable tuple
                neighbor_id = tuple(node)
                print(point_id, neighbor_id, weight_distance)
                self.complex.add_edge(point_id, neighbor_id, weight_distance)

    def remove_point(self, point):
        point_id = tuple(point)
        assert self.complex.has_node(point_id), 'Cannot remove point not already in simplex'
        self.complex.remove_node(point)

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

    def get_simplex(self, point, dim=1):
        if type(point) != tuple:
            point = tuple(point)
        if not self.complex.has_node(point):
            return None
        neighbors = self.complex.neighbors(point)
        candidate_simplicies = combinations(neighbors, dim)
        #TODO Finish functino definintion after defining simplex class
        pass

    def is_fully_connected(self, nodes):
        edges = combinations(nodes,2)
        for edge in edges:
            if not self.complex.has_edge(*edge):
                return False
        return True
