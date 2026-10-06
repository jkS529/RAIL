class DisjointSet:
    """Union-find that tracks the number of nodes in components containing infected nodes."""

    def __init__(self, graph_size, infect_list):
        self.union_set = list(range(graph_size))
        self.rank_count = [1] * graph_size
        self.infect_Component_Status = infect_list.copy()
        # Number of nodes in components that contain at least one infected node
        self.CCDScore = 0

    def find_root(self, node):
        if node != self.union_set[node]:
            root_node = self.find_root(self.union_set[node])
            self.union_set[node] = root_node
        return self.union_set[node]

    def merge(self, node1, node2):
        root1 = self.find_root(node1)
        root2 = self.find_root(node2)

        if root1 != root2:
            rank1 = self.rank_count[root1]
            rank2 = self.rank_count[root2]
            infect_Status_node1Root = self.infect_Component_Status[root1]
            infect_Status_node2Root = self.infect_Component_Status[root2]

            # A clean component merged into an infected one becomes at risk
            if infect_Status_node1Root == 1 and infect_Status_node2Root == 0:
                self.CCDScore = self.CCDScore + rank2
            if infect_Status_node1Root == 0 and infect_Status_node2Root == 1:
                self.CCDScore = self.CCDScore + rank1

            if infect_Status_node1Root == 1 or infect_Status_node2Root == 1:
                self.infect_Component_Status[root1] = 1
                self.infect_Component_Status[root2] = 1

            # Union by size
            if self.rank_count[root2] > self.rank_count[root1]:
                self.union_set[root1] = root2
                self.rank_count[root2] += self.rank_count[root1]
            else:
                self.union_set[root2] = root1
                self.rank_count[root1] += self.rank_count[root2]

    def get_rank(self, root_node: int) -> int:
        """Size of the component rooted at root_node, or 0 if it contains no infected node."""
        if self.infect_Component_Status[root_node] == 0:
            return 0
        return self.rank_count[root_node]
