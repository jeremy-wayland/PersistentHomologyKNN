import numpy as np
import networkx as nx
from simplicial_complex import SimplicialComplex
from collections import Counter


class PHKnn:

    def __init__(self,
                 radius,
                 k,
                 graph_distance_metric=None,
                 weight_distance_metric=None):
        self.radius = radius
        self.k = k
        self.graph_distance_metric = graph_distance_metric
        self.weight_distance_metric = weight_distance_metric
        self.sc = None

    def fit(self, X, y):
        assert isinstance(X, np.ndarray),\
            f'Data matrix must be a numpy array, got {type(X)}'
        assert isinstance(y, np.ndarray), \
            f'Target labels must be given as a numpy array, got {type(y)}'
        self.sc = SimplicialComplex(self.radius,
                                    self.graph_distance_metric,
                                    self.weight_distance_metric,
                                    vertices=X)
        attributes = {}
        for data, label in zip(X, y):
            attributes[tuple(data)] = {'label': label}
        nx.set_node_attributes(self.sc.complex, attributes)

    def predict(self, X):
        assert isinstance(X, np.ndarray),\
            f'Data matrix must be a numpy array of shape (n_queries,n_features)'\
            f', got {type(X)}'
        assert self.sc is not None, 'Attempting to predict before fitting to training data'
        if len(X.shape) == 1:
            X.reshape(1, X.shape[0])
        predictions = []
        #carry out knn algorithm for each query in matrix
        for x in X:
            neighbors = self.get_nearest_k_neighbors(x)
            prediction = self.majority_vote(neighbors)
            predictions.append(prediction)
        return predictions

    def get_sorted_neighbors(self, point):
        point = tuple(point)
        self.sc.add_point(point)
        distances = []
        for node in self.sc.complex.nodes:
            if nx.has_path(self.sc.complex, node, point):
                distance = nx.shortest_path_length(self.sc.complex,
                                                   source=point,
                                                   target=node,
                                                   weight='weight')
                distances.append((node, distance))
        self.sc.remove_point(point)
        distances.sort(key=lambda l: l[1])
        neighbors = [pair[0] for pair in distances]
        return neighbors

    def get_nearest_k_neighbors(self, point):
        sorted_neighbors = self.get_sorted_neighbors(point)
        # check that there are enough neighbors to conduct classification
        if len(sorted_neighbors) < self.k:
            message = f'Only {len(sorted_neighbors)} neighbors to point {point} '\
                f'found, at least {self.k} needed for classification.\n '\
                f'This indicates that the data matrix '\
                f'is too sparse near {point} for a successful classification'
            print(message)
            return None
        nearest_neighbors = sorted_neighbors[:self.k]
        return nearest_neighbors

    def get_labels(self,nodes):
        if nodes is None:
            return None
        all_labels = nx.get_node_attributes(self.sc.complex, 'label')
        neighbor_labels = []
        for n in nodes:
            label = all_labels[n]
            neighbor_labels.append(label)
        return neighbor_labels

    def majority_vote(self, neighbors):
        labels = self.get_labels(neighbors)
        if labels is None:
            return None
        vote_count = Counter(labels)
        winning_vote = vote_count.most_common(1)[0]
        return winning_vote
















