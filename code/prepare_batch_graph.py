import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict, deque
from typing import List


class SparseMatrix:
    def __init__(self):
        self.row_index = []
        self.col_index = []
        self.value = []
        self.row_num = 0
        self.col_num = 0


class PrepareBatchGraph:
    """Pack a batch of graphs into the sparse matrices fed to the network.

    Two views are built for each graph:
      - global: all non-covered nodes with an uncovered edge, used for node indexing
        and auxiliary features;
      - risk: the same nodes restricted to the subgraph reachable from infected nodes,
        on which the GNN is run.
    """

    def setup_train(self, idxes, g_list, covered, actions):
        return self._setup_risk_graph_input(idxes, g_list, covered, actions)

    def setup_pred_all(self, idxes, g_list, covered):
        return self._setup_risk_graph_input(idxes, g_list, covered, None)

    def _setup_risk_graph_input(self, idxes, g_list, covered, actions):
        global_result = self._setup_graph_input(idxes, g_list, covered, actions, with_structure=False)

        # Treat nodes outside the risk-reachable set as covered
        risk_covered = list(covered)
        for idx in idxes:
            covered_set = set(covered[idx])
            risk_set = self.compute_risk_reachable(g_list[idx], covered_set)
            rc = covered_set.copy()
            for node in g_list[idx].nodes():
                if node not in risk_set:
                    rc.add(node)
            risk_covered[idx] = rc

        risk_result = self._setup_graph_input(idxes, g_list, risk_covered, actions, with_structure=True)

        risk_align = self._build_risk_align(
            idxes, g_list,
            global_result['idx_map_list'],
            risk_result['idx_map_list'])

        return {
            'rep_global': global_result['rep_global'],
            'aux_input': global_result['aux_input'],
            'idx_map_list': global_result['idx_map_list'],
            'action_select_risk': risk_result['action_select'],
            'rep_global_risk': risk_result['rep_global'],
            'n2nsum_param_risk': risk_result['n2nsum_param'],
            'laplacian_param_risk': risk_result['laplacian_param'],
            'subgsum_param_risk': risk_result['subgsum_param'],
            'node_feat_risk': risk_result['node_feat'],
            'graph_feat_risk': risk_result['graph_feat'],
            'risk_align': self.convert_sparse_to_tensor(risk_align),
        }

    def _setup_graph_input(self, idxes: np.ndarray, g_list: List[nx.Graph], covered, actions, with_structure):
        act_select = SparseMatrix()
        rep_global = SparseMatrix()
        aux_feat = []
        node_feat = []
        graph_feat = []
        idx_map_list = [[] for _ in idxes]
        avail_act_cnt = [0 for _ in idxes]
        node_cnt = 0

        for i, idx in enumerate(idxes):
            temp_feat = []

            graph = g_list[idx]
            node_num = graph.number_of_nodes()
            edge_num = graph.number_of_edges()

            if node_num > 0:
                temp_feat.append(len(covered[idx]) / node_num)
            else:
                temp_feat.append(0.0)

            # idx_map[j] >= 0 iff node j is non-covered and has an uncovered edge
            curr_avail_act_cnt, counter, twohop_number, idx_map = self.get_status_info(graph, covered[idx])
            idx_map_list[i] = idx_map
            avail_act_cnt[i] = curr_avail_act_cnt
            infect_list = graph.graph.get('infect_list')
            infect_avail_cnt = 0
            if infect_list is not None:
                for node_id, mapped in enumerate(idx_map):
                    if mapped >= 0 and infect_list[node_id] != 0:
                        infect_avail_cnt += 1
            infect_ratio = infect_avail_cnt / curr_avail_act_cnt if curr_avail_act_cnt > 0 else 0.0
            graph_feat.append([1.0, infect_ratio])

            if edge_num > 0:
                temp_feat.append(counter / edge_num)
            else:
                temp_feat.append(0.0)

            if node_num > 0:
                temp_feat.append(twohop_number / (node_num * node_num))
            else:
                temp_feat.append(0.0)
            temp_feat.append(1.0)

            node_cnt += avail_act_cnt[i]
            aux_feat.append(temp_feat)

        inner_graph = {
            'in_edges': defaultdict(list),
            'subgraph': defaultdict(list),
            'num_nodes': node_cnt,
            'num_edges': 0,
            'num_subgraph': len(idxes),
        }

        if actions is not None:
            act_select.row_num = len(idxes)
            act_select.col_num = node_cnt
        else:
            rep_global.row_num = node_cnt
            rep_global.col_num = len(idxes)

        node_cnt = 0
        edge_cnt = 0
        for i, idx in enumerate(idxes):
            graph = g_list[idx]
            infect_list = graph.graph.get('infect_list')
            idx_map = idx_map_list[i].copy()
            t = 0
            for j in range(graph.number_of_nodes()):
                if idx_map[j] < 0:
                    continue
                idx_map[j] = t
                infect_flag = 0.0
                if infect_list is not None and infect_list[j] != 0:
                    infect_flag = 1.0
                node_feat.append([1.0, infect_flag])
                inner_graph['subgraph'][i].append(node_cnt + t)

                if actions is None:
                    rep_global.row_index.append(node_cnt + t)
                    rep_global.col_index.append(i)
                    rep_global.value.append(1.0)
                t += 1
            assert t == avail_act_cnt[i]
            if actions is not None:
                act = actions[idx]
                act_select.row_index.append(i)
                act_select.col_index.append(node_cnt + idx_map[act])
                act_select.value.append(1.0)

            if with_structure:
                for u, v in graph.edges():
                    if idx_map[u] < 0 or idx_map[v] < 0:
                        continue
                    x = idx_map[u] + node_cnt
                    y = idx_map[v] + node_cnt
                    self.add_edge_to_inner_graph(inner_graph, edge_cnt, x, y)
                    edge_cnt += 1
                    self.add_edge_to_inner_graph(inner_graph, edge_cnt, y, x)
                    edge_cnt += 1
            node_cnt += avail_act_cnt[i]

        result = {
            'action_select': self.convert_sparse_to_tensor(act_select),
            'rep_global': self.convert_sparse_to_tensor(rep_global),
            'aux_input': aux_feat,
            'node_feat': node_feat,
            'graph_feat': graph_feat,
            'idx_map_list': idx_map_list,
        }
        if with_structure:
            n2nsum_param, laplacian_param = self.n2n_construct(inner_graph)
            result['n2nsum_param'] = self.convert_sparse_to_tensor(n2nsum_param)
            result['laplacian_param'] = self.convert_sparse_to_tensor(laplacian_param)
            result['subgsum_param'] = self.convert_sparse_to_tensor(self.subg_construct(inner_graph))
        return result

    def add_edge_to_inner_graph(self, inner_graph, idx, first, second):
        inner_graph['in_edges'][second].append([idx, first])
        inner_graph['num_edges'] += 1
        assert inner_graph['num_edges'] - 1 == idx

    def n2n_construct(self, inner_graph):
        """Sum-aggregation adjacency matrix and the graph Laplacian."""
        num_nodes = inner_graph['num_nodes']
        result = SparseMatrix()
        result.row_num = num_nodes
        result.col_num = num_nodes

        result_laplacian = SparseMatrix()
        result_laplacian.row_num = num_nodes
        result_laplacian.col_num = num_nodes

        for i in range(num_nodes):
            edge_list = inner_graph['in_edges'][i]

            if len(edge_list) > 0:
                result_laplacian.value.append(len(edge_list))
                result_laplacian.row_index.append(i)
                result_laplacian.col_index.append(i)

            for edge in edge_list:
                result.value.append(1.0)
                result.row_index.append(i)
                result.col_index.append(edge[1])

                result_laplacian.value.append(-1.0)
                result_laplacian.row_index.append(i)
                result_laplacian.col_index.append(edge[1])

        return result, result_laplacian

    def subg_construct(self, inner_graph):
        """Matrix that sums node embeddings of each graph in the batch."""
        result = SparseMatrix()
        result.row_num = inner_graph['num_subgraph']
        result.col_num = inner_graph['num_nodes']

        for i in range(inner_graph['num_subgraph']):
            for node in inner_graph['subgraph'][i]:
                result.value.append(1.0)
                result.row_index.append(i)
                result.col_index.append(node)

        return result

    def get_status_info(self, graph, covered):
        idx_map = [-1] * graph.number_of_nodes()
        counter = 0
        twohop_number = 0
        node_twohop_counter = {}
        curr_avail_act_cnt = 0
        for u, v in graph.edges():
            if u in covered or v in covered:
                counter += 1
            else:
                if idx_map[u] < 0:
                    curr_avail_act_cnt += 1
                if idx_map[v] < 0:
                    curr_avail_act_cnt += 1
                idx_map[u] = 0
                idx_map[v] = 0
            if u in node_twohop_counter:
                twohop_number += node_twohop_counter[u]
                node_twohop_counter[u] += 1
            else:
                node_twohop_counter[u] = 1

            if v in node_twohop_counter:
                twohop_number += node_twohop_counter[v]
                node_twohop_counter[v] += 1
            else:
                node_twohop_counter[v] = 1

        return curr_avail_act_cnt, counter, twohop_number, idx_map

    @staticmethod
    def convert_sparse_to_tensor(sparse_matrix):
        indices = np.matrix([sparse_matrix.row_index, sparse_matrix.col_index]).T
        return tf.compat.v1.SparseTensorValue(indices, sparse_matrix.value,
                                              (sparse_matrix.row_num, sparse_matrix.col_num))

    @staticmethod
    def compute_risk_reachable(graph, covered):
        """BFS from infected nodes to find risk-reachable non-covered nodes."""
        infect_list = graph.graph.get('infect_list')
        covered_set = set(covered) if not isinstance(covered, set) else covered
        if infect_list is None:
            return set(n for n in graph.nodes() if n not in covered_set)
        visited = set()
        queue = deque()
        for node in graph.nodes():
            if infect_list[node] == 1 and node not in covered_set:
                queue.append(node)
                visited.add(node)
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if v not in covered_set and v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited

    def _build_risk_align(self, idxes, g_list, idx_map_global_list, idx_map_risk_list):
        """Sparse matrix [node_cnt_global, node_cnt_risk] mapping risk nodes to global positions."""
        risk_align = SparseMatrix()
        avail_global = []
        avail_risk = []
        for i, idx in enumerate(idxes):
            avail_global.append(sum(1 for x in idx_map_global_list[i] if x >= 0))
            avail_risk.append(sum(1 for x in idx_map_risk_list[i] if x >= 0))
        risk_align.row_num = sum(avail_global)
        risk_align.col_num = sum(avail_risk)

        global_offset = 0
        risk_offset = 0
        for i, idx in enumerate(idxes):
            g_seq = 0
            r_seq = 0
            for j in range(g_list[idx].number_of_nodes()):
                in_global = idx_map_global_list[i][j] >= 0
                in_risk = idx_map_risk_list[i][j] >= 0
                if in_global and in_risk:
                    risk_align.row_index.append(global_offset + g_seq)
                    risk_align.col_index.append(risk_offset + r_seq)
                    risk_align.value.append(1.0)
                if in_global:
                    g_seq += 1
                if in_risk:
                    r_seq += 1
            global_offset += avail_global[i]
            risk_offset += avail_risk[i]
        return risk_align
