import networkx as nx
from typing import List
from collections import defaultdict
from disjoint_set import DisjointSet


def add_edge(current_adj, node0, node1):
    current_adj[node0].append(node1)
    current_adj[node1].append(node0)


def recover_add_node(adj_list_backup, node_flags, current_adj, node, disjoint_set):
    """Add a removed node back to the graph and connect it to already-recovered neighbors."""
    if disjoint_set.infect_Component_Status[node] == 1:
        disjoint_set.CCDScore += 1
    neighbors = adj_list_backup[node]
    for neighbor_node in neighbors:
        if node_flags[neighbor_node]:
            add_edge(current_adj, node, neighbor_node)
            disjoint_set.merge(node, neighbor_node)
    node_flags[node] = True


def get_robustness(graph: nx.Graph, solution: List[int]) -> float:
    """Evaluate a removal sequence by adding nodes back in reverse order.

    Returns the accumulated score and the per-step score curve.
    """
    num_graph_nodes = graph.number_of_nodes()

    score_list = []
    original_graph_adj = {n: list(graph.neighbors(n)) for n in graph.nodes}
    current_adj = defaultdict(list)
    disjoint_set = DisjointSet(num_graph_nodes, graph.graph['infect_list'])
    node_flags = [False] * num_graph_nodes

    total_score = 0.0
    temp = 0.0
    norm = num_graph_nodes * (num_graph_nodes - 1) / 2.0

    for node in reversed(solution):
        recover_add_node(original_graph_adj, node_flags, current_adj, node, disjoint_set)
        normalized_score = disjoint_set.CCDScore / norm

        total_score += normalized_score
        score_list.append(normalized_score)
        temp = normalized_score
    # Exclude the step where all nodes are present
    total_score -= temp
    score_list.reverse()
    return total_score / len(graph), score_list
