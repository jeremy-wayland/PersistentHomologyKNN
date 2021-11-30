import numpy as np
import pandas as pd
import networkx as nx

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
        self.points = []
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
            self.points.append(point)
            dist = self.get_distance_vector(point)
            for i in range(len(dist)-1):
                if dist[i][0]<=2*self.radius:
                    #convert points to hashable tuples
                    point_id, neighbor_id = tuple(point), tuple(self.points[i])
                    self.complex.add_edge(point_id,neighbor_id, weight=dist[i][1])

    def remove_point(self, point):
        assert point in self.points, 'Cannot remove point not already in simplex'
        self.complex.remove_node(point)
        self.points.remove(point)

    def get_distance_vector(self, point):
        graph_distance = \
        [self.graph_distance_metric(point,x) for x in self.points]
        if self.graph_distance_metric == self.weight_distance_metric:
            weight_distance = graph_distance
        else:
            weight_distance = \
            [self.weight_distance_metric(point,x) for x in self.points]
        distances = zip(graph_distance, weight_distance)
        return np.array(list(distances))


    def import_vertices(self, vertices, delim=','):
        assert type(vertices) in [str, list, np.ndarray, pd.DataFrame],\
            'Vertices input of incorrect type, must be filepath, Pandas Dataframe\
            or Numpy ndarray'
        if type(vertices)==str:
            with open(vertices) as f:
                return np.loadtxt(f, delimiter=delim)
        else:
            return np.array(vertices)
