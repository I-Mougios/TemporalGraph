from typing import List, Iterable, Dict, Set, Tuple
from itertools import takewhile, chain, combinations
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx
from helpers import Edge
from datetime import datetime


class Graph:
    """
    The Graph class represents a temporal subgraph, capturing a subset of edges that occur within a specified
    time interval. It provides functionalities for analyzing the structure and centrality of this subgraph.

    __init__ Method
    Description:
    Initializes a Graph instance, representing a subgraph of temporal edges within a specified time interval.
    Filters the provided edges_list to include only edges whose timestamps fall within the specified interval.
    Creates internal data structures to store the edges, vertices, adjacency list, and other graph-related information.

    Parameters:
    edges_list (List[Edge]): A list of Edge objects, each representing an edge in the graph with attributes source,
    target, and timestamp.
    interval_start (int): The inclusive start of the time interval. Only edges with a timestamp greater than or equal
    to this value will be included in the subgraph.
    interval_end (int): The exclusive end of the time interval. Only edges with a timestamp strictly less than this
    value will be included in the subgraph.

    Attributes (Read-Only):

    __edges (List[Edge]): The list of edges that belong to the subgraph, filtered based on their timestamp to fall
      within the interval [interval_start, interval_end).

    __vertices (set): The set of unique target vertices from the edges within the subgraph.

    __all_vertices (List): The set of all unique vertices, considering both source and target nodes of the edges.

    __interval_start (int): The inclusive start of the time interval for the subgraph.

    __interval_end (int): The exclusive end of the time interval for the subgraph.

    __adjacency_list (Dict[str, List[str]]): The adjacency list representation of the subgraph, where nodes map to
      their neighboring nodes.

    __adjacency_matrix (pd.DataFrame or None): The adjacency matrix representation of the subgraph.
      Initially None, it's computed on demand from the adjacency list.

    __reachable_neighbors (Dict[str, List[str]]): A dictionary where keys are all starting vertices of edges and values
      are their reachable neighbors.

    __shortest_paths (Dict[str, List[str]]): A dictionary where keys are source vertices and values are the first
      encountered shortest paths to their reachable neighbors.

    __all_shortest_paths (Dict[str, List[List[str]]]): A dictionary where keys are source vertices and values are lists
      containing all possible shortest paths to their reachable neighbors.

    __number_of_vertices (int): The number of unique sources vertices in the graph.

    Properties

    edges (property):
    Description: Returns a copy of the edges list to prevent direct modification.
    Returns: A list of Edge objects representing the edges in the subgraph.

    vertices (property):
    Description: Returns a set of unique target vertices from the edges within the subgraph.
    Returns: A set of string values representing the unique target vertices.

    all_vertices (property):
    Description: Returns a list of all unique vertices, considering both source and target nodes of the edges.
    Returns: A list of string values representing all unique vertices.

    number_of_vertices (property):
    Returns the number of unique vertices in the subgraph.
    If not cached, it computes the number from the vertices property.

    interval_start (property):
    Description: Returns the inclusive start of the time interval for the subgraph.
    Returns: An integer representing the start of the time interval.

    interval_end (property):
    Description: Returns the exclusive end of the time interval for the subgraph.
    Returns: An integer representing the end of the time interval.

    adjacency_list (property):
    Description: Returns a copy of the adjacency list to prevent direct modification.
    Returns: A dictionary where keys are nodes and values are lists of neighboring nodes.

    adjacency_matrix (property):
    Description: Returns the adjacency matrix representation of the subgraph.
    If not computed, it's constructed based on the adjacency list.
    Returns: A pandas DataFrame representing the adjacency matrix.

    nodes_indegree (property):

    Description: Returns a dictionary where keys are nodes and values are their indegree (incoming edges).
    Returns: A dictionary mapping nodes to their indegree values.

    nodes_outdegree (property):
    Description: Returns a dictionary where keys are nodes and values are their outdegree (outgoing edges).
    Returns: A dictionary mapping nodes to their outdegree values.

    nodes_degree (property):
    Description: Returns a dictionary where keys are nodes and values are their total degree
                 (incoming and outgoing edges).
    Returns: A dictionary mapping nodes to their total degree values.

    reachable_neighbors (property):
    Description: Returns a dictionary where keys are starting vertices and values are lists of their reachable
    neighbors.
    Returns: A dictionary mapping nodes to lists of reachable nodes.

    shortest_paths (property):
    Description: Returns a dictionary where each vertex is mapped to the shortest paths from that vertex to all
    other vertices. If multiple shortest paths exist between a pair of vertices,
    only the first encountered path is returned.
    Returns: A dictionary where keys are vertices and values are dictionaries containing
    shortest paths to other vertices.

    all_shortest_paths (property):
    Description: Returns a dictionary where each vertex is mapped to the shortest paths from that vertex to all other
    vertices.
     This property handles cases where multiple shortest paths exist between a pair of vertices, storing all such paths.
    Returns: A dictionary where keys are vertices and values are dictionaries
    containing lists of all possible shortest paths to other vertices.

    Special Methods Implemented:
        __len__ Method
        Description:
        Returns the number of edges in the temporal subgraph.
        Returns:
        An integer representing the number of edges.

        __getitem__ Method
        Description:
        Returns an edge from the subgraph's edges list at the specified index.
        Parameters:
        index (int): The index of the edge to retrieve.
        Returns:
        An Edge object representing the edge at the given index.

        __setitem__ Method
        Description:
        Prevents direct modification of edges in the subgraph. Raises an AttributeError if called.
        Raises:
        AttributeError: If an attempt is made to modify edges directly.

        __repr__ Method
        Description:
        Returns a formal string representation of the Graph instance.
        Returns:
        A string representation of the graph, including the edges list, interval start, and interval end.

        __str__ Method
        Description:
        Returns a user-friendly string representation of the edges in the subgraph.
        Returns:
        A string representing the edges in the subgraph.

        append(self, value: Edge)
        Description:
        Adds a new edge to the subgraph and updates the relevant data structures.
        Parameters:
        value (Edge): The edge to be added to the subgraph.
        Side Effects:
        Modifies the __edges, __adjacency_list, __adjacency_matrix, __vertices, __reachable_neighbors, __shortest_paths,
         __all_shortest_paths, and __number_of_vertices attributes to reflect the changes made by adding the new edge.

        insert(self, index: int, value: Edge)
        Description:
        Inserts a new edge at the specified index in the subgraph's edges list and updates the relevant data structures.
        Parameters:
        index (int): The index where the new edge should be inserted.
        value (Edge): The edge to be inserted into the subgraph.
        Side Effects:
        Modifies the __edges, __adjacency_list, __adjacency_matrix, __vertices, __reachable_neighbors, __shortest_paths,
        __all_shortest_paths, and __number_of_vertices attributes to reflect the changes made by inserting the new edge.

        remove(self, value: Edge)
        Description:
        Removes the specified edge from the subgraph and updates the relevant data structures.
        Parameters:
        value (Edge): The edge to be removed from the subgraph.
        Side Effects:
        Modifies the __edges, __adjacency_list, __adjacency_matrix, __vertices, __reachable_neighbors, __shortest_paths,
        __all_shortest_paths, and __number_of_vertices attributes to reflect the changes made by removing the edge.
        Raises:
        ValueError: If the specified edge is not found in the subgraph's edges list.

        extend(self, values: Iterable[Edge])
        Description:
        Extends the subgraph's edges list by adding multiple edges at once and updates the relevant data structures.
        Parameters:
        values (Iterable[Edge]): An iterable containing the edges to be added to the subgraph.
        Side Effects:
        Modifies the __edges, __adjacency_list, __adjacency_matrix, __vertices, __reachable_neighbors, __shortest_paths,
        __all_shortest_paths, and __number_of_vertices attributes to reflect the changes made by adding the new edges.

        sort_edges(self, reverse: bool = False)
        Description:
        Sorts the edges in the subgraph based on their timestamps.
        Parameters:
        reverse (bool, optional): If True, sorts the edges in descending order of timestamp.
        Defaults to False (ascending order).
        Side Effects:
        Modifies the __edges attribute to reflect the sorted order of the edges.

        Helper and Private Methods
        _find_reachable_nodes(self, node: str) -> List[str]
        Description:

        Identifies all nodes reachable from a given node in the graph using a breadth-first search (BFS) approach.
        Prevents infinite loops by tracking visited nodes and avoiding revisiting them.
        Parameters:

        node (str): The starting node from which to find reachable nodes.
        Returns:

        A list of nodes that are reachable from the given node.
        _find_shortest_path(self, vertex: str) -> Dict[str, List[str]]
        Description:
        Computes the shortest paths from a given vertex to all its reachable neighbors using BFS.
        Stores the shortest paths in a dictionary.
        Parameters:
        vertex (str): The starting vertex from which to find shortest paths.
        Returns:
        A dictionary where keys are reachable nodes and values are lists representing the shortest path from the
         starting vertex to that node.

        _find_all_shortest_paths(self, vertex: str) -> Dict[str, List[List[str]]]
        Description:
        Computes all shortest paths from a given vertex to its reachable neighbors using BFS.
        Handles cases where multiple shortest paths exist between a pair of vertices.
        Parameters:
        vertex (str): The starting vertex from which to find shortest paths.
        Returns:
        A dictionary where keys are reachable nodes and values are lists of all possible shortest paths from the
        starting vertex to that node.

        _calculate_closeness(self, vertex: str, _type: str = 'both') -> float
        Description:
        Calculates the closeness centrality for a given vertex in the graph.
        Filters shortest paths based on the specified _type parameter (in, out, or both).
        Computes the closeness centrality score using the formula: (N-1) / sum of distances.
        Parameters:
        vertex (str): The vertex for which to calculate closeness centrality.
        _type (str, optional): The type of closeness centrality to calculate (default is 'both').
        Returns:
        A float representing the closeness centrality score for the vertex.

        Analysis Methods
        These methods provide various functionalities for analyzing the structure and properties of the graph.

        Visualization
        plot(self) -> None: Creates a visual representation of the graph using NetworkX and matplotlib libraries.
        It assumes the graph is directed.


        Centrality Measures
        degree_centrality(self) -> pd.DataFrame: Calculates the degree centrality for each vertex in the graph and
        returns a DataFrame summarizing the results. Degree centrality measures the number of connections a vertex has
        with other vertices.

        plot_degree_centrality(self) -> plt.Figure: Creates a bar chart to visualize the distribution of
        degree centrality scores across all vertices.

        closeness_centrality(self, _type='both') -> Dict[str, float]: Calculates the closeness centrality for each
        vertex in the graph based on the specified type ('in', 'out', or 'both').
        Closeness centrality measures how close a vertex is to all other reachable nodes.

        plot_closeness_centrality(self, _type='both'): Creates a histogram to visualize the distribution of closeness
        centrality scores for all vertices.

        betweenness_centrality(self) -> Dict[str, int]: Calculates the betweenness centrality for each vertex in the
        graph. Betweenness centrality measures how often a vertex appears on the shortest paths between other pairs of
        vertices, indicating its importance within the network.

        plot_betweenness_centrality(self): Creates a histogram to visualize the distribution of betweenness centrality
        scores for all vertices.

        eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]: Calculates the
        eigenvector centrality for each vertex in the graph. Eigenvector centrality measures a node's importance based
        on its connections to other important nodes.

        plot_eigenvector_centrality(self): Creates a histogram to visualize the distribution of eigenvector centrality
        scores for all vertices.

        katz_centrality(self, alpha: float = 0.1, beta: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        Calculates the Katz centrality for each vertex in the graph. Similar to eigenvector centrality, it considers the
        importance of a node's neighbors and the length of the connections (walks) in the graph.

        plot_katz_centrality(self): Creates a histogram to visualize the distribution of Katz centrality scores for all
        vertices.

        Similarity Analysis
        sdg_for_a_set_of_vertices(self, vertices: Set[str]) -> pd.DataFrame: Computes the similarity matrix based on the
        shortest geodesic distance (SGD) for a given set of vertices. Similarity is defined as the negative of the
        shortest path length between two vertices.

        get_neighbors_from_adjacency_matrix(self, vertex) -> Set[str]: Retrieves all neighbors
         (both incoming and outgoing) of a given vertex from the adjacency matrix.

         intersection(self, other) -> Tuple[Set[str], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
         Description:
        This method computes the intersection between the current graph and another graph
        (both instances of the Graph class).
        The intersection consists of:

        Common vertices between the two graphs.
        Edges from the current graph that are between these common vertices.
        Edges from the other graph that are between these common vertices.
        Parameters:
        other: An instance of the Graph class with which to find the intersection.
        Returns:
        A tuple containing:
        vertices_intersection: A set of vertices common to both graphs.
        edges_from_self: A set of edges (as tuples of source and target) from the current graph where both vertices are
        in the common vertex set.
        edges_from_other: A set of edges (as tuples of source and target) from the other graph where both vertices are
        in the common vertex set.
        Raises:
        TypeError: If the input other is not an instance of the Graph class.

    """

    def __init__(self, edges_list: List[Edge], interval_start: int, interval_end: int):
        """
        Initializes a Graph instance, which represents a subgraph of temporal edges occurring within a
         specified time interval.

        Parameters:
            edges_list (List[Edge]):
                A list of `edge` objects, each representing an edge in the graph which has attributes source, target
                timestamp.
            interval_start (int):
                The inclusive start of the time interval. Only edges with a `timestamp` greater than or equal to this
                value will be included in the subgraph.
            interval_end (int):
                The exclusive end of the time interval. Only edges with a `timestamp` strictly less than this value will
                 be included in the subgraph.
        """
        self.__edges = list(takewhile(lambda edge:
                                      interval_start <= edge.timestamp < interval_end,
                                      edges_list))

        self.__interval_start = interval_start
        self.__interval_end = interval_end
        self.__adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.__adjacency_matrix: None | pd.DataFrame = None
        self.__vertices: Set[str] | None = None
        self.__all_vertices: List[str] | None = None
        self.__reachable_neighbors: Dict[str, List[str]] | None = None
        self.__shortest_paths: Dict[str, Dict[str, List[str]]] | None = None
        self.__all_shortest_paths: Dict[str, Dict[str, List[List[str]]]] | None = None
        self.__number_of_vertices: int | None = None

        # Initialize the adjacency list
        self._create_adjacency_list()

    def __len__(self):
        return len(self.__edges)

    def __getitem__(self, index):
        return self.__edges[index]

    def __setitem__(self, index: int | slice, value: Edge | List[Edge]):
        raise AttributeError("Cannot modify edges directly. Use append, insert, remove or extend methods.")

    def __repr__(self):
        return (f'Graph(edges_list={self.__edges},'
                f' interval_start={self.__interval_start},'
                f' interval_end={self.__interval_end})')

    def __str__(self):
        return f'{self.edges}'

    @property
    def edges(self) -> List[Edge]:
        """
        Description:
           Returns a copy of the list of edges in the graph to prevent modifications to the internal edge data.
        """
        return self.__edges.copy()  # Return a copy to prevent modification

    @property
    def vertices(self) -> Set[str]:
        """
        Description:
         Returns a copy of the set of all unique target vertices from the edges.
         If the set is not initialized, it builds the set from the edges.
        """
        if self.__vertices:
            return self.__vertices.copy()
        else:
            self.__vertices = {edge.target for edge in self.__edges}
            return self.__vertices.copy()

    @property
    def all_vertices(self) -> List[str]:
        """
        Description:
         Returns a list of all unique vertices in the graph, considering both the source and target vertices.
         If the list is not initialized, it creates the list by iterating through the edges.

        """
        if self.__all_vertices:
            return self.__all_vertices.copy()
        else:
            vertices = []
            for edge in self:
                if edge.source not in vertices:
                    vertices.append(edge.source)
                if edge.target not in vertices:
                    vertices.append(edge.target)
            self.__all_vertices = vertices
            return self.__all_vertices.copy()

    @property
    def number_of_vertices(self) -> int:
        """
        Description:
        Returns the number of unique vertices in the graph.
        If not initialized, it computes the number from the vertices property.
        """
        if self.__number_of_vertices:
            return self.__number_of_vertices
        else:
            self.__number_of_vertices = len(self.vertices)
            return self.__number_of_vertices

    @property
    def interval_start(self) -> int:
        """
         Description:
         Returns the end of the time interval for the graph.
        """
        return self.__interval_start

    @property
    def interval_end(self) -> int:
        """
        Description:
         Returns the start of the time interval for the graph.
        """
        return self.__interval_end

    @property
    def adjacency_list(self) -> Dict[str, List[str]]:
        """
        Description:
        Returns a copy of the adjacency list representing the graph.
        If the adjacency list is not initialized, it calls a function to create it from the edges.
        """
        if self.__adjacency_list:
            return self.__adjacency_list.copy()
        else:
            return self._create_adjacency_list().copy()

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """
        Description:
         Returns a copy of the adjacency matrix representing the graph.
        If not initialized, it generates the matrix from the adjacency list.
        """
        if self.__adjacency_matrix is not None and not self.__adjacency_matrix.empty:
            return self.__adjacency_matrix.copy()
        else:
            self._adjacency_list_to_matrix()
            return self.__adjacency_matrix.copy()

    @property
    def nodes_indegree(self) -> pd.Series:
        """How many directed edges terminate to each node"""
        # Sum of the columns of adjacency matrix
        return self.adjacency_matrix.sum(axis=0).rename('Degree')

    @property
    def nodes_outdegree(self) -> pd.Series:
        """How many directed edges originate to each node"""
        # Sum the values by row of adjacency matrix
        return self.adjacency_matrix.sum(axis=1).rename('Degree')

    @property
    def nodes_degree(self) -> pd.DataFrame:
        """Number of edges incident to each node"""
        return (self.nodes_outdegree + self.nodes_indegree).rename('Degree')

    @property
    def reachable_neighbors(self) -> Dict[str, List[str]]:
        """
        Description:
         Returns a copy of a dictionary where each vertex is mapped to a list of its reachable neighbors, excluding the
         vertex itself.
         If the dictionary is not initialized, it computes the reachable neighbors for each vertex by calling a helper
         function _find_reachable_nodes. If the vertex is present in its own reachable neighbors, it is removed.
        """
        if self.__reachable_neighbors:
            return self.__reachable_neighbors.copy()
        else:
            self.__reachable_neighbors = {}
            for vertex in self.adjacency_list:
                self.__reachable_neighbors[vertex] = self._find_reachable_nodes(vertex)
                try:
                    self.__reachable_neighbors[vertex].remove(vertex)
                except ValueError:
                    pass
            return self.__reachable_neighbors.copy()

    @property
    def shortest_paths(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Description:
            Returns a copy of a dictionary where each vertex is mapped to the shortest paths from that vertex to all other
            vertices.

        CAVEAT:
            In case of multiple shortest paths between a pair of vertices, it will return only the first encountered path.

        If the dictionary is not initialized, it computes the shortest paths by calling a helper function
        _find_shortest_path for each vertex in the adjacency list.
        """
        if self.__shortest_paths:
            return self.__shortest_paths.copy()
        else:
            self.__shortest_paths = {}
            for vertex in self.adjacency_list:
                self.__shortest_paths[vertex] = self._find_shortest_path(vertex)
            return self.__shortest_paths.copy()

    @property
    def all_shortest_paths(self) -> Dict[str, Dict[str, List[List[str]]]]:
        """
        Description:
           Returns a copy of a dictionary where each vertex is mapped to the shortest paths from that vertex to all other
           vertices. It handles cases where multiple shortest paths between a pair of vertices exist.

        If the dictionary is not initialized, it computes the shortest paths by calling a helper function
       _find_shortest_path for each vertex in the adjacency list.
       """
        if self.__all_shortest_paths:
            return self.__all_shortest_paths.copy()
        else:
            self.__all_shortest_paths = {}
            for vertex in self.adjacency_list:
                self.__all_shortest_paths[vertex] = self._find_all_shortest_paths(vertex)
            return self.__all_shortest_paths.copy()

    def append(self, value: Edge):
        self.__edges.append(value)
        self._append_adjacency_list(value)
        self.__adjacency_matrix = None
        self.__vertices = None
        self.__reachable_neighbors = None
        self.__shortest_paths = None
        self.__all_shortest_paths = None
        self.__number_of_vertices = None

    def insert(self, index: int, value: Edge):
        self.__edges.insert(index, value)
        self._append_adjacency_list(value)
        self.__adjacency_matrix = None
        self.__vertices = None
        self.__reachable_neighbors = None
        self.__shortest_paths = None
        self.__all_shortest_paths = None
        self.__number_of_vertices = None

    def remove(self, value: Edge):
        try:
            self.__edges.remove(value)
            self._remove_from_adjacency_list(value)
            self.__adjacency_matrix = None
            self.__vertices = None
            self.__reachable_neighbors = None
            self.__shortest_paths = None
            self.__all_shortest_paths = None
            self.__number_of_vertices = None
        except ValueError:
            print(f'{value} not found in set of edges')

    def extend(self, values: Iterable[Edge]):
        self.__edges.extend(values)
        for value in values:
            self._append_adjacency_list(value)

        self.__adjacency_matrix = None
        self.__vertices = None
        self.__reachable_neighbors = None
        self.__shortest_paths = None
        self.__all_shortest_paths = None
        self.__number_of_vertices = None

    def sort_edges(self, reverse: bool = False):
        """Sort the edges based on the timestamp."""
        self.__edges.sort(key=lambda edge: edge.timestamp, reverse=reverse)

    def intersection(self, other) -> Tuple[Set[str],
                                           Set[Tuple[str, str]],
                                           Set[Tuple[str, str]]
                                          ]:
        """
        Description:
        This method computes the intersection between the current graph and another graph (both instances of the Graph class).
        The intersection consists of:
        - Common vertices between the two graphs.
        - Edges from the current graph that are between these common vertices.
        - Edges from the other graph that are between these common vertices.

        Parameters:
        - other: An instance of the Graph class with which to find the intersection.

        Returns:
        - A tuple containing:
            1. vertices_intersection: A set of vertices common to both graphs.
            2. edges_from_self: A set of edges (as tuples of source and target) from the current graph
               where both vertices are in the common vertex set.
            3. edges_from_other: A set of edges (as tuples of source and target) from the other graph
               where both vertices are in the common vertex set.

        Raises:
        - TypeError: If the input `other` is not an instance of the Graph class.
         """

        if isinstance(other, Graph):
            vertices_intersection = set(self.all_vertices).intersection(set(other.all_vertices))
            edges_from_self = set((edge.source, edge.target)
                                  for edge in self
                                  if edge.source in vertices_intersection and edge.target in vertices_intersection
                                  )

            edges_from_other = set((edge.source, edge.target)
                                   for edge in other
                                   if edge.source in vertices_intersection and edge.target in vertices_intersection
                                   )

            return vertices_intersection, edges_from_self, edges_from_other
        else:
            raise TypeError('The input must be an instance of Graph')

    def _create_adjacency_list(self):
        for edge in self.__edges:
            self.__adjacency_list[edge.source].append(edge.target)
            # self.__adjacency_list[edge.target].append(edge.source)

        return self.adjacency_list

    def _append_adjacency_list(self, edge: Edge):
        self.__adjacency_list[edge.source].append(edge.target)
        # self.__adjacency_list[edge.target].append(edge.source)

    def _remove_from_adjacency_list(self, edge: Edge):
        self.__adjacency_list[edge.source].remove(edge.target)
        # self.__adjacency_list[edge.target].remove(edge.source)

    def _adjacency_list_to_matrix(self):
        # Get the total number of vertices (assume nodes are labeled from 0 to n-1)
        n = len(self.all_vertices)

        # Initialize an nxn matrix with zeros
        adj_matrix = np.zeros((n, n), dtype=int)

        adj_matrix = pd.DataFrame(data=adj_matrix,
                                  index=[node for node in self.__all_vertices],
                                  columns=[node for node in self.__all_vertices])

        # Fill in the matrix based on the adjacency list
        for key, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                adj_matrix.loc[key, neighbor] = 1  # Set 1 where there's an edge

        self.__adjacency_matrix = adj_matrix.fillna(0).astype(int)

    def _append_all_vertices(self, edge: Edge):
        if edge.source not in self.__all_vertices:
            self.__all_vertices.append(edge.source)
        if edge.target not in self.__all_vertices:
            self.__all_vertices.append(edge.target)

    def interval_end_to_datetime(self):
        return datetime.fromtimestamp(self.__interval_end).date()

    def plot(self) -> None:
        """
        Description:
        Visualize the graph's structure. It assumes that the graph is directed.
        """
        g = nx.DiGraph(self.adjacency_list)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes((0, 0, 1, 1))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=120, font_size=6, ax=ax)
        plt.title('Directed Graph')

    def degree_centrality(self):
        counter = defaultdict(int)
        for value in self.nodes_degree.to_dict().values():
            counter[value] += 1

        df = pd.DataFrame(sorted(counter.items(), key=lambda x: x[0]),
                          columns=['Degree', 'Frequency Count'])
        df['Relative Frequency'] = df['Frequency Count'] / df['Frequency Count'].sum()
        return df

    def plot_degree_centrality(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Frequency Distribution of Degree Centrality')
        return sns.barplot(x='Degree', y='Relative Frequency',
                           data=self.degree_centrality(), ax=ax)

    def _find_reachable_nodes(self, node: str) -> List[str]:
        """

        This method identifies all nodes reachable from a given node in a graph, represented by an adjacency list.
        It uses a breadth-first search-like approach to explore nodes and employs a set to prevent
        infinite loops caused by cycles.

        Parameters:

        node:
        The starting node from which to find all reachable nodes. It should be a key in the adjacency_list attribute
        of the graph.
        Returns:

        reachable_nodes:
        A list of nodes that are reachable from the given node. This includes direct neighbors and nodes that can be
        reached through paths from the given node.
        Key Points:

        Preventing Infinite Loops:
        The set reached_nodes ensures that each node is processed only once, preventing the algorithm from getting stuck
        in cycles and avoiding infinite loops in cyclic graphs.
        Method Logic:

        Initialize Reachable Nodes:
        The method starts by initializing reachable_nodes with the direct neighbors of the given node. It accesses
        these neighbors using the adjacency_list.
        Track Reached Nodes:
        A set called reached_nodes is used to track the nodes whose neighbors have already been processed.
        This prevents revisiting nodes and infinite looping, especially in cyclic graphs.
        Neighbor Exploration:
        For each neighbor in reachable_nodes:
        If the neighbor has not been added to reached_nodes, it is added to avoid revisiting.
        The method then finds additional neighbors of this node that haven't been added to reachable_nodes yet,
        and extends reachable_nodes with these new nodes.
        Termination:
        The process continues until no new neighbors can be added. The final list of reachable nodes is returned.

        """
        # Neighbors are reachable nodes
        reachable_nodes = [neighbor
                           for neighbor in self.adjacency_list[node]
                           if neighbor != node]
        # To keep track the nodes for which I have already added their neighbors
        reached_nodes = set()
        for neighbor in reachable_nodes:
            if neighbor not in reached_nodes:
                reached_nodes.add(neighbor)
                # the node itself can be added via a neighbor though but only one time
                new_reachable_nodes = [neigh
                                       for neigh in self.adjacency_list[neighbor]
                                       if neigh not in reachable_nodes]
                reachable_nodes.extend(new_reachable_nodes)
        return reachable_nodes

    def _find_shortest_path(self, vertex: str):
        """
           Find the shortest paths from the vertex under study to all its reachable neighbors using
           BFS(Breadth First Search).

           Parameters:
           - The adjacency list of the graph.
           - Reachable_neighbors: Dictionary where keys are nodes and values are lists of reachable neighbors.
           - start_node: The node from which to find the shortest paths.

           Returns:
           - shortest_paths: Dictionary where keys are nodes and values are lists representing the shortest path from
            start_node to that node.
           """
        # Initialize a dictionary that stores the shortest path to each reachable node of the vertex under study
        shortest_paths = {node: [] for node in self.reachable_neighbors[vertex]}

        # Dictionary that stores distances from the vertex under study
        distances = dict.fromkeys(self.reachable_neighbors[vertex], float('inf'))
        distances[vertex] = 0

        # Initialize a deque, it will store (node, path) tuples
        queue = deque(
                      [(vertex, [vertex])]  # list of one tuple
                      )
        while queue:
            current_node, path = queue.popleft()
            # Explore the neighbors of the vertex under study
            for neighbor in self.adjacency_list[current_node]:
                # If the neighbor is reachable and has not been visited (distance is still infinity)
                if distances[neighbor] == float('inf'):
                    # Updating the distances for all the neighbors of the start vertex, then for the neighbors of
                    # neighbors and so on. If a distance is not updated will be inf and will evaluate True
                    # on the if statement above
                    distances[neighbor] = distances[current_node] + 1  # Update the distance
                    new_path = path + [neighbor]  # Extend the path to include this neighbor

                    # Store the shortest path to this neighbor
                    shortest_paths[neighbor] = new_path

                    # Add the neighbor to the queue for further exploration
                    queue.append((neighbor, new_path))

        return shortest_paths

    def _find_all_shortest_paths(self, vertex: str):
        """
        Finds all shortest paths from the given vertex to its reachable neighbors using BFS (Breadth First Search).

        This function explores the graph starting from the specified vertex and calculates all possible shortest paths
        to each reachable neighbor. If multiple shortest paths exist, all of them will be stored in the result.

        Parameters:
        - vertex: The starting node from which to compute the shortest paths.

        Returns:
        - all_paths: A dictionary where the keys are reachable nodes, and the values are lists of paths (each path
          represented as a list of nodes) from the start vertex to that node.

        Algorithm:
        - The BFS ensures that all shortest paths are found by exploring nodes layer by layer, updating the distance
          for each node and appending new shortest paths when a node is reached through multiple paths of equal length.
        """
        # Initialize a dictionary to store all shortest paths for each reachable node
        all_paths = {node: [] for node in self.reachable_neighbors[vertex]}

        # Dictionary to track the shortest distance from the start vertex to each neighbor
        distances = dict.fromkeys(self.reachable_neighbors[vertex], float('inf'))

        # Initialize the BFS queue with the starting vertex and the initial path (just the vertex itself)
        queue = deque([(vertex, [vertex])])

        # Set the distance from the start vertex to itself as 0
        distances[vertex] = 0

        while queue:
            current, path = queue.popleft()

            # Explore the neighbors of the current node
            for neighbor in self.adjacency_list.get(current, []):
                # Case 1: Visiting this neighbor for the first time
                if distances[neighbor] == float('inf'):
                    # Set the shortest distance to this neighbor
                    distances[neighbor] = distances[current] + 1

                    # Create a new path extending the current path with the neighbor
                    new_path = path + [neighbor]

                    # Store this as the first shortest path to the neighbor
                    all_paths[neighbor] = [new_path]

                    # Add the neighbor and its path to the queue for further exploration
                    queue.append((neighbor, new_path))

                # Case 2: Neighbor has already been reached, but we might find another shortest path
                elif distances[neighbor] == distances[current] + 1:
                    # In this case, the `neighbor` has already been reached via another path of the same length.
                    # This indicates that `current` is another valid predecessor to `neighbor`, meaning we found
                    # a second (or more) shortest path to `neighbor`.

                    # **Key Explanation**:
                    # - `distances[neighbor] == distances[current] + 1` ensures that `neighbor` is being reached from a
                    #   different predecessor (`current`), but the shortest path length is the same.
                    # - We only add another path if this condition holds, meaning `neighbor` is reachable from different
                    #   nodes but at the same shortest path distance.
                    # Create another new path by extending the current path
                    new_path = path + [neighbor]

                    # Append the new shortest path to the list of paths for this neighbor
                    all_paths[neighbor].append(new_path)

                    # Add the neighbor to the queue for continued exploration (if necessary)
                    queue.append((neighbor, new_path))

        return all_paths

    def flattened_shortest_paths(self):
        """
        Description:
        This method flattens the nested structure of all the shortest paths in the graph into a single list. It retrieves
        all the shortest paths between nodes in the graph and concatenates them into one comprehensive list for easier access.

        Functionality:
        - The method first accesses the dictionary `all_shortest_paths` which contains all possible shortest paths between
          each pair of nodes.
        - The `all_shortest_paths` is structured as a dictionary where the keys are the destination nodes, and the values
          are lists of shortest paths to those nodes from any starting node.
        - The method iterates through each entry in the dictionary, extracts the lists of shortest paths, and appends each
          individual path to a new list called `paths`.

        Returns:
        - A list of paths, where each path is a list of nodes representing the shortest path between two vertices.

        Example:
        If the graph has the following shortest paths:
        - From A to B: ['A', 'C', 'B']
        - From A to D: ['A', 'D']
        - From B to C: ['B', 'C']

        The method will return: [['A', 'C', 'B'], ['A', 'D'], ['B', 'C']]
    """

        shortest_paths = self.all_shortest_paths.values()
        paths = []
        for shortest_paths_to_node in shortest_paths:
            for list_of_shortest_paths in shortest_paths_to_node.values():
                for path in list_of_shortest_paths:
                    paths.append(path)

        return paths

    def _calculate_closeness(self, vertex: str, _type: str = 'both') -> float:
        """
        Description:
        This method calculates the closeness centrality for a given vertex in the graph. Closeness centrality measures
        how close a vertex is to all other reachable vertices, based on the sum of the shortest path distances.

        Parameters:
        - vertex (str): The node for which to calculate closeness centrality.
        - type (str, optional): Defines whether to calculate "in", "out", or "both" types of closeness centrality:
            - 'in': Measures how close the vertex is to all vertices that can reach it.
            - 'out': Measures how close the vertex is to all vertices it can reach.
            - 'both': Considers both incoming and outgoing paths (default).

        Functionality:
        - Filters the shortest paths where the vertex is either the start (out_paths) or the end (in_paths).
        - Depending on the `type` parameter, the method sums the distances of the selected paths.
        - Closeness centrality is computed as the reciprocal of the sum of distances, normalized by the number of vertices.

        Returns:
        - float: Closeness centrality score for the vertex. Returns 0 if the vertex is isolated or unreachable.
        """
        flattened_shortest_paths = self.flattened_shortest_paths()
        # Paths where the vertex can reach other vertices
        out_paths = filter(lambda path: path[0] == vertex, flattened_shortest_paths)

        # Paths where the vertex is reached by other vertices
        in_paths = filter(lambda path: path[-1] == vertex, flattened_shortest_paths)

        if _type == 'in':
            distances = map(lambda path: len(path) - 1, in_paths)
        elif _type == 'out':
            distances = map(lambda path: len(path) - 1, out_paths)
        else:  # type == 'both'
            distances = map(lambda path: len(path) - 1, chain(in_paths, out_paths))

        total_distance = sum(distances)

        # Return 0 if the total distance is 0 (no reachable vertices)
        if total_distance == 0:
            return 0

        # Closeness centrality formula: (N-1) / sum of distances
        return (self.number_of_vertices - 1) / total_distance

    def closeness_centrality(self, _type='both') -> Dict[str, float]:
        """
        This method iterates over all vertices in the graph, calculates their closeness centrality
         using _calculate_closeness(), and stores the result in a dictionary.
        Each value is rounded to four decimal places for precision.
        """
        closeness_centrality = {}
        for vertex in set(self.all_vertices):
            closeness_centrality[vertex] = round(self._calculate_closeness(vertex, _type=_type), 4)
        return closeness_centrality

    def plot_closeness_centrality(self, _type='both'):
        to_plot = self.closeness_centrality(_type=_type).values()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(to_plot, density=True, bins=100)
        ax.set_title('Frequency Distribution of Closeness Centrality')

    def betweenness_centrality(self) -> Dict[str, int]:
        """
        Description:

        This method calculates the betweenness centrality of each vertex in the graph and returns the results as a
        pandas DataFrame. Betweenness centrality measures how often a node appears on the shortest paths between other
        pairs of vertices, which can indicate its importance or influence within the network.

        Functionality:

        Get a list with all shortest paths using the self.flattened_shortest_paths() method.
        Chain the list of paths into a sequence of vertices and count the number of how often they appear.
        """
        # Get the paths in one list
        flattened_shortest_paths = self.flattened_shortest_paths()
        sequence_of_vertices = chain.from_iterable(flattened_shortest_paths)

        # Count the frequency of each vertex appearing in the shortest paths( Betweenness Score of the vertices)
        counter = defaultdict(int)
        for vertex in sequence_of_vertices:
            counter[vertex] += 1

        return counter

    def plot_betweenness_centrality(self):
        to_plot = self.betweenness_centrality().values()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(to_plot, density=True, bins=100)
        ax.set_title('Frequency Distribution of Betweenness Centrality')

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """
        Description:

        This method calculate the eigenvector centrality for each vertex in the graph. which measures the importance 
        of a node in a graph based on its connections to other important nodes. A node with high eigenvector centrality 
        is connected to many other nodes that themselves have high centrality, capturing both direct and indirect 
        influence within the network.

        Parameters:
        - max_iter: Maximum number of iterations (default is 100)
        - tol: Tolerance for convergence (default is 1e-6)

        Returns:
        - A dictionary where keys are vertex names and values are the eigenvector centrality scores.
        """
        adjacency_matrix = np.array(self.adjacency_matrix)
        n = adjacency_matrix.shape[0]
        centrality = np.ones(n)
        for _ in range(max_iter):
            new_centrality = adjacency_matrix.dot(centrality)
            new_centrality /= np.linalg.norm(new_centrality)
            if np.linalg.norm(new_centrality - centrality) < tol:
                break
            centrality = new_centrality
        vertices = self.all_vertices
        eigenvector_centrality = {vertex: round(centrality[i], 4) for i, vertex in enumerate(vertices)}

        return eigenvector_centrality
    
    def plot_eigenvector_centrality(self):
        to_plot = self.eigenvector_centrality().values()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(to_plot, density=True, bins=100)
        ax.set_title('Frequency Distribution of Eigenvector Centrality')

    def katz_centrality(self, alpha: float = 0.1, beta: float = 1.0,
                        max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """
        Description:

        This method calculate the Katz centrality for each vertex in the graph. It is similar to Eigenvector Centrality but 
        includes a parameter a = attenuation factor to account for the length of walks in the graph and an optional parameter 
        b that allows a baseline score for each node. The centrality calculation considers not only the importance of a node's 
        neighbors but also the length of the connections (walks), allowing even distant nodes to contribute to a node's 
        centrality, albeit with decreasing influence as the distance grows.

        Parameters:
        - alpha: Attenuation factor (default is 0.1)
        - beta: Baseline centrality (default is 1.0)
        - max_iter: Maximum number of iterations (default is 100)
        - tol: Tolerance for convergence (default is 1e-6)

        Returns:
        - A dictionary where keys are vertex names and values are the Katz centrality scores.
        """
        adjacency_matrix = np.array(self.adjacency_matrix)
        n = adjacency_matrix.shape[0]
        centrality = np.ones(n) * beta
        for _ in range(max_iter):
            new_centrality = alpha * adjacency_matrix.dot(centrality) + beta
            if np.linalg.norm(new_centrality - centrality) < tol:
                break
            centrality = new_centrality
        vertices = self.all_vertices
        katz_centrality = {vertex: round(centrality[i], 4) for i, vertex in enumerate(vertices)}

        return katz_centrality

    def plot_katz_centrality(self):
        to_plot = self.katz_centrality().values()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(to_plot, density=True, bins=100)
        ax.set_title('Frequency Distribution of Katz Centrality')

    def sdg_for_a_set_of_vertices(self, vertices: Set[str]) -> pd.DataFrame:
        """
        Description:
        This method computes the similarity matrix based on the shortest geodesic distance (SGD)
        for a given set of vertices. The similarity score between two vertices (u, v) is defined as:

            SGD(u, v) = -d_geodesic(u, v)

        where d_geodesic(u, v) is the length of the shortest path between vertices u and v.
        The score is negative to indicate that smaller distances mean higher similarity.

        Parameters:
        - vertices: A set of vertices for which the similarity matrix is to be computed.

        Returns:
        - similarity_matrix: A dictionary representing the geodesic similarity for the given set of vertices.
          Each entry (u, v) in the matrix contains the similarity score based on the shortest path between u and v.

        Assumptions:
        - The graph is represented as an adjacency list, and shortest paths between all node pairs are
          precomputed and stored in a dictionary.
        - If there is no path between two vertices, the entry in the similarity matrix remains 0.
        """

        # Ensure vertices are a set (to remove duplicates)
        vertices = set(vertices)

        # Retrieve all shortest paths (flattened format) from the precomputed dictionary
        all_shortest_paths = self.flattened_shortest_paths()

        # Dictionary to store the similarity scores (shortest paths as negative values)
        shortest_path_dict = dict.fromkeys(combinations(vertices, r=2), -np.inf)

        # Iterate over all precomputed paths and filter those whose start and end nodes are in the given set of vertices
        for path in all_shortest_paths:
            if path[0] in vertices and path[-1] in vertices:
                # Calculate similarity score: the negative of the path length (since length is number of steps)
                score = -len(path) + 1
                shortest_path_dict[(path[0], path[-1])] = score

        # # Convert the set of vertices to a list for indexing purposes in the matrix
        # vertices_list = list(vertices)
        #
        # # Initialize a similarity matrix of zeros, indexed and labeled by the vertices
        # similarity_matrix = pd.DataFrame(data=np.zeros((len(vertices), len(vertices))),
        #                                  index=vertices_list,
        #                                  columns=vertices_list)
        #
        # # Populate the matrix with similarity scores from the shortest_path_dict
        # for key, value in shortest_path_dict.items():
        #     start_node, end_node = key[0], key[1]
        #     similarity_matrix.loc[start_node, end_node] = value

        return shortest_path_dict

    def get_neighbors_from_adjacency_matrix(self, vertex) -> Set[str]:
        """
        Description:
        This method retrieves all the neighbors of a given vertex from the adjacency matrix.
        Neighbors are vertices directly connected to the input vertex by an edge, either incoming or outgoing.

        Parameters:
        - vertex: The vertex for which to find neighbors.

        Returns:
        - A set of vertices that are neighbors (either in or out) of the given vertex.
        """
        # Retrieve neighbors by checking both rows (outgoing edges) and columns (incoming edges) in the adjacency matrix
        return (set(self.adjacency_matrix.loc[:, vertex][self.adjacency_matrix.loc[:, vertex] == 1].index) |
                set(self.adjacency_matrix.loc[vertex, :][self.adjacency_matrix.loc[vertex, :] == 1].index))
