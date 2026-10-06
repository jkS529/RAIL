import networkx as nx
import numpy as np
from collections import deque
from disjoint_set import DisjointSet


class RiskEnv:
    """Environment for sequential node removal on a graph with infected nodes.

    Infected nodes are given by g.graph['infect_list'] and can never be removed.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.num_covered_edges = 0
        self.covered_set = set()
        self.action_list = []
        self.state_seq = []
        self.act_seq = []
        self.reward_seq = []
        self.sum_rewards = []

    def s0_init_with_graph(self, g):
        self.graph = g
        self.covered_set.clear()
        self.action_list.clear()
        self.num_covered_edges = 0
        self.state_seq.clear()
        self.act_seq.clear()
        self.reward_seq.clear()
        self.sum_rewards.clear()

    def step(self, action_node):
        """Remove a node and return the reward."""
        self.state_seq.append(self.action_list.copy())
        self.act_seq.append(action_node)
        self.step_without_reward(action_node)

        reward = self.get_reward()
        self.reward_seq.append(reward)
        self.sum_rewards.append(reward)
        return reward

    def step_without_reward(self, action_node):
        """Remove a node without computing the reward."""
        assert self.graph is not None, "Graph is not initialized."
        assert action_node not in self.covered_set, "Node is already covered."
        infect_list = self.graph.graph.get('infect_list')
        if infect_list is not None:
            assert 0 <= action_node < len(infect_list), "Action index out of range."
            assert infect_list[action_node] == 0, "Cannot select infected node."

        self.covered_set.add(action_node)
        self.action_list.append(action_node)

        for neigh in self.graph.neighbors(action_node):
            if neigh not in self.covered_set:
                self.num_covered_edges += 1

    def get_risk_reachable_set(self):
        """Non-covered nodes reachable from at least one infected node."""
        infect_list = self.graph.graph.get('infect_list')
        if infect_list is None:
            return set(n for n in self.graph.nodes if n not in self.covered_set)
        visited = set()
        queue = deque()
        for node in self.graph.nodes:
            if infect_list[node] == 1 and node not in self.covered_set:
                queue.append(node)
                visited.add(node)
        while queue:
            u = queue.popleft()
            for v in self.graph.neighbors(u):
                if v not in self.covered_set and v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited

    def has_available_risk_action(self):
        """Whether an uninfected risk-reachable node with an uncovered edge remains."""
        risk_set = self.get_risk_reachable_set()
        infect_list = self.graph.graph.get('infect_list')
        for node in risk_set:
            if node in self.covered_set:
                continue
            if infect_list is not None and infect_list[node] != 0:
                continue
            for neigh in self.graph.neighbors(node):
                if neigh not in self.covered_set:
                    return True
        return False

    def random_action(self, risk_set=None):
        """Randomly select an uninfected node with at least one uncovered edge."""
        assert self.graph is not None, "Graph is not initialized."

        avail_list = []
        for node in self.graph.nodes:
            if (node not in self.covered_set) and (self.graph.graph['infect_list'][node] == 0):
                if risk_set is not None and node not in risk_set:
                    continue
                for neigh in self.graph.neighbors(node):
                    if neigh not in self.covered_set:
                        avail_list.append(node)
                        break

        assert len(avail_list) > 0, "No available actions."
        return np.random.choice(avail_list)

    def is_terminal(self):
        """All edges are covered except those between infected nodes."""
        assert self.graph is not None, "Graph is not initialized."
        infect_edges_set = set()
        infect_index = [i for i, value in enumerate(self.graph.graph['infect_list']) if value != 0]

        for node in infect_index:
            for neigh in self.graph.neighbors(node):
                if neigh in infect_index:
                    infect_edges_set.add((min(node, neigh), max(node, neigh)))
        return self.num_covered_edges == (self.graph.number_of_edges() - len(infect_edges_set))

    def get_reward(self):
        return -(self.get_remaining_cnd_score() / self.graph.number_of_nodes())

    def get_remaining_cnd_score(self):
        """Number of non-covered nodes in components that contain an infected node."""
        disjoint_set = DisjointSet(self.graph.number_of_nodes(), self.graph.graph['infect_list'])
        for node in self.graph.nodes:
            if node not in self.covered_set:
                for neigh in self.graph.neighbors(node):
                    if neigh not in self.covered_set:
                        disjoint_set.merge(node, neigh)

        root_ids = set()
        for node in self.graph.nodes:
            root_ids.add(disjoint_set.union_set[node])

        score = 0.0
        for node in root_ids:
            score += disjoint_set.get_rank(node)
        return score
