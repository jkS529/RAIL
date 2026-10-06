import tensorflow.compat.v1 as tf
import numpy as np
import networkx as nx
import random
import time
import sys
import os
from tqdm import tqdm

import graph_utils
from replay_mem import NStepReplayMem
from env import RiskEnv
from prepare_batch_graph import PrepareBatchGraph
tf.disable_eager_execution()


def HXA(g, method):
    """Adaptive degree (HDA) / betweenness (HBA) baseline that never removes infected nodes."""
    sol = []
    G = g.copy()
    infect_list = g.graph.get('infect_list')
    while nx.number_of_edges(G) > 0:
        if method == 'HDA':
            dc = nx.degree_centrality(G)
        elif method == 'HBA':
            dc = nx.betweenness_centrality(G)
        keys = list(dc.keys())
        values = list(dc.values())
        if infect_list is not None:
            for i, k in enumerate(keys):
                if infect_list[int(k)] == 1:
                    values[i] = -float('inf')
        maxTag = np.argmax(values)
        if values[maxTag] == -float('inf'):
            break
        node = keys[maxTag]
        sol.append(int(node))
        G.remove_node(node)
    solution = sol + list(set(g.nodes()) ^ set(sol))
    solutions = [int(i) for i in solution]
    Robustness, _ = graph_utils.get_robustness(g, solutions)
    return Robustness, sol


def my_argMax(scores):
    n = len(scores)
    pos = -1
    best = -10000000
    for i in range(n):
        if pos == -1 or scores[i] > best:
            pos = i
            best = scores[i]
    return pos


SEED = 42
# Hyper Parameters:
GAMMA = 1.0                 # discount factor
UPDATE_TIME = 1000          # target network update interval
EMBEDDING_SIZE = 64
MAX_ITERATION = 100000
LEARNING_RATE = 0.0001
MEMORY_SIZE = 500000
Alpha = 0.001               # weight of the structural regularization loss
N_STEP = 5
NUM_MIN = 30                # training graph size range
NUM_MAX = 50
INFECT_RATIO = 0.2          # fraction of infected nodes in training graphs
REG_HIDDEN = 32
BATCH_SIZE = 64
initialization_stddev = 0.01
n_valid = 200               # number of validation graphs
aux_dim = 4
num_env = 10
max_bp_iter = 3             # number of GNN propagation layers


class RAIL:

    def __init__(self):
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.TrainSet = {}
        self.TestSet = {}
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list = []
        self.g_list = []
        self.reg_hidden = REG_HIDDEN

        self.nStepReplayMem = NStepReplayMem(MEMORY_SIZE, seed=SEED)

        for i in range(num_env):
            self.env_list.append(RiskEnv())
            self.g_list.append(nx.Graph())
        self.test_env = RiskEnv()

        # Rows: non-covered nodes of each graph; used to align Q-values with node ids
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")

        # Risk subgraph (nodes reachable from infected nodes)
        self.action_select_risk = tf.sparse_placeholder(tf.float32, name="action_select_risk")
        self.rep_global_risk = tf.sparse_placeholder(tf.float32, name="rep_global_risk")
        self.n2nsum_param_risk = tf.sparse_placeholder(tf.float32, name="n2nsum_param_risk")
        self.laplacian_param_risk = tf.sparse_placeholder(tf.float32, name="laplacian_param_risk")
        self.subgsum_param_risk = tf.sparse_placeholder(tf.float32, name="subgsum_param_risk")
        self.node_feat_risk = tf.placeholder(tf.float32, [None, 2], name="node_feat_risk")
        self.graph_feat_risk = tf.placeholder(tf.float32, [None, 2], name="graph_feat_risk")
        self.risk_align = tf.sparse_placeholder(tf.float32, name="risk_align")

        self.target = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name="target")
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")

        # Online Q network and target Q network
        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildNet('q_online')
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.BuildNet('q_target')

        self.copyTargetQNetworkOperation = [a.assign(b) for a, b in zip(self.Q_param_listT, self.Q_param_list)]
        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)

        self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    # ======================== Network ========================

    @staticmethod
    def _build_gnn(node_input, graph_input, n2nsum_param, subgsum_param, w_n2l, p_node_conv, p_node_conv2, p_node_conv3):
        """Message-passing encoder with sum aggregation.

        Returns node embeddings [node_cnt, embed] and graph embeddings [batch, embed].
        """
        input_message = tf.matmul(tf.cast(node_input, tf.float32), w_n2l)
        input_potential_layer = tf.nn.relu(input_message)

        y_input_message = tf.matmul(tf.cast(graph_input, tf.float32), w_n2l)
        y_input_potential_layer = tf.nn.relu(y_input_message)

        cur_message_layer = tf.nn.l2_normalize(input_potential_layer, axis=1)
        y_cur_message_layer = tf.nn.l2_normalize(y_input_potential_layer, axis=1)

        for lv in range(max_bp_iter):
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(n2nsum_param, tf.float32), cur_message_layer)
            node_linear = tf.matmul(n2npool, p_node_conv)

            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(subgsum_param, tf.float32), cur_message_layer)
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)

            cur_msg_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
            merged = tf.concat([node_linear, cur_msg_linear], 1)
            cur_message_layer = tf.nn.relu(tf.matmul(merged, p_node_conv3))

            y_cur_msg_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
            y_merged = tf.concat([y_node_linear, y_cur_msg_linear], 1)
            y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        return cur_message_layer, y_cur_message_layer

    @staticmethod
    def _bilinear(action_embed, graph_summary, cross_product):
        """(action_embed outer graph_summary) . cross_product -> [N, embed]"""
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),
                         tf.expand_dims(graph_summary, axis=1))
        Shape = tf.shape(action_embed)
        result = tf.reshape(
            tf.matmul(temp,
                      tf.reshape(tf.tile(cross_product, [Shape[0], 1]),
                                 [Shape[0], Shape[1], 1])),
            Shape)
        return result

    def BuildNet(self, scope):
        with tf.variable_scope(scope):
            # Encoder weights (variable creation order must match saved checkpoints)
            w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev))
            p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev))
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev))
            p_node_conv3 = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.embedding_size], stddev=initialization_stddev))

            # Decoder weights
            cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev))
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev))
            last_w = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev))

            cur_msg_risk, y_msg_risk = self._build_gnn(
                self.node_feat_risk, self.graph_feat_risk, self.n2nsum_param_risk, self.subgsum_param_risk,
                w_n2l, p_node_conv, p_node_conv2, p_node_conv3)

            # ---- Q(s, a) for the sampled actions (training) ----
            action_embed = tf.sparse_tensor_dense_matmul(
                tf.cast(self.action_select_risk, tf.float32), cur_msg_risk)   # [batch, embed]
            embed_s_a = self._bilinear(action_embed, y_msg_risk, cross_product)  # [batch, embed]

            last_output = tf.nn.relu(tf.matmul(embed_s_a, h1_weight))       # [batch, reg_hidden]
            last_output = tf.concat([last_output, self.aux_input], 1)       # [batch, reg_hidden + aux_dim]
            q_pred = tf.matmul(last_output, last_w)                         # [batch, 1]

            # ---- Structural regularization ----
            loss_recons = 2 * tf.trace(tf.matmul(
                tf.transpose(cur_msg_risk),
                tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param_risk, tf.float32), cur_msg_risk)))
            edge_num = tf.maximum(tf.sparse_reduce_sum(tf.cast(self.n2nsum_param_risk, tf.float32)), 1.0)
            loss_recons = tf.divide(loss_recons, edge_num)

            loss_rl = tf.losses.mean_squared_error(self.target, q_pred)
            loss = loss_rl + Alpha * loss_recons
            trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # ---- Q(s, ·) for all candidate nodes (prediction) ----
            # risk_align maps risk-subgraph nodes back to global node positions
            risk_node_aligned = tf.sparse_tensor_dense_matmul(
                tf.cast(self.risk_align, tf.float32), cur_msg_risk)             # [node_g, embed]
            rep_y_risk = tf.sparse_tensor_dense_matmul(
                tf.cast(self.rep_global_risk, tf.float32), y_msg_risk)          # [node_r, embed]
            risk_rep_aligned = tf.sparse_tensor_dense_matmul(
                tf.cast(self.risk_align, tf.float32), rep_y_risk)               # [node_g, embed]
            embed_s_a_all = self._bilinear(risk_node_aligned, risk_rep_aligned, cross_product)  # [node_g, embed]

            last_output = tf.nn.relu(tf.matmul(embed_s_a_all, h1_weight))   # [node_g, reg_hidden]
            rep_aux = tf.sparse_tensor_dense_matmul(
                tf.cast(self.rep_global, tf.float32), self.aux_input)       # [node_g, aux_dim]
            last_output = tf.concat([last_output, rep_aux], 1)
            q_on_all = tf.matmul(last_output, last_w)                       # [node_g, 1]

        q_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return loss, trainStep, q_pred, q_on_all, q_params

    # ======================== Graph generation ========================

    def gen_graph(self, num_min, num_max):
        cur_n = np.random.randint(num_max - num_min + 1) + num_min
        g = nx.barabasi_albert_graph(n=cur_n, m=4)

        infect_list = np.zeros(cur_n, dtype=int)
        num_infected = int(cur_n * INFECT_RATIO)
        infected_indices = np.random.choice(cur_n, num_infected, replace=False)
        infect_list[infected_indices] = 1
        g.graph['infect_list'] = infect_list
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        self.ClearTrainGraphs()
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.clear()

    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            assert t not in self.TestSet
            self.TestSet[t] = g
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            assert t not in self.TrainSet
            self.TrainSet[t] = g

    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        result_degree = 0.0
        result_betweeness = 0.0
        for i in tqdm(range(n_valid)):
            new_graph = self.gen_graph(NUM_MIN, NUM_MAX)
            val_degree, sol = HXA(new_graph, 'HDA')
            result_degree += val_degree
            val_betweenness, sol = HXA(new_graph, 'HBA')
            result_betweeness += val_betweenness
            self.InsertGraph(new_graph, is_test=True)
        print('Validation of HDA: %.6f' % (result_degree / n_valid))
        print('Validation of HBA: %.6f' % (result_betweeness / n_valid))
        self.baseline_hda = result_degree / n_valid
        self.baseline_hba = result_betweeness / n_valid

    # ======================== Simulator ========================

    def Run_simulator(self, num_seq, eps, TrainSet, n_step=N_STEP):
        """Play episodes with eps-greedy actions restricted to the risk-reachable set."""
        num_env = len(self.env_list)
        n = 0
        while n < num_seq:
            for i in range(num_env):
                is_end = (self.env_list[i].graph.number_of_nodes() == 0
                          or self.env_list[i].is_terminal()
                          or not self.env_list[i].has_available_risk_action())
                if is_end:
                    if self.env_list[i].graph.number_of_nodes() > 0:
                        n = n + 1
                        self.nStepReplayMem.add_env(self.env_list[i], n_step)
                    g_sample = random.choice(TrainSet)
                    self.env_list[i].s0_init_with_graph(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break
            Random = False
            if random.uniform(0, 1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if Random:
                    risk_set = self.env_list[i].get_risk_reachable_set()
                    action_node = self.env_list[i].random_action(risk_set=risk_set)
                else:
                    action_node = my_argMax(pred[i])
                self.env_list[i].step(action_node)

    # ======================== Prediction ========================

    def Do_Predict(self, g_list, covered, isSnapShot):
        n_graphs = len(g_list)
        pred = []
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = min(BATCH_SIZE, n_graphs - i)
            batch_idxes = np.arange(i, i + bsize, dtype=np.int32)
            batch_pred = self._predict_batch(batch_idxes, g_list, covered, isSnapShot)
            pred.extend(batch_pred)
        return pred

    def _predict_batch(self, batch_idxes, g_list, covered, is_snapshot):
        inputs = PrepareBatchGraph().setup_pred_all(batch_idxes, g_list, covered)
        feed_dict = {
            self.rep_global: inputs['rep_global'],
            self.aux_input: np.array(inputs['aux_input']),
            self.rep_global_risk: inputs['rep_global_risk'],
            self.n2nsum_param_risk: inputs['n2nsum_param_risk'],
            self.laplacian_param_risk: inputs['laplacian_param_risk'],
            self.subgsum_param_risk: inputs['subgsum_param_risk'],
            self.node_feat_risk: np.array(inputs['node_feat_risk'], dtype=np.float32).reshape((-1, 2)),
            self.graph_feat_risk: np.array(inputs['graph_feat_risk'], dtype=np.float32).reshape((-1, 2)),
            self.risk_align: inputs['risk_align'],
        }

        if is_snapshot:
            raw_output = self.session.run(self.q_on_allT, feed_dict=feed_dict)
        else:
            raw_output = self.session.run(self.q_on_all, feed_dict=feed_dict)

        raw = np.array(raw_output).reshape(-1)
        idx_map_list = inputs['idx_map_list']

        pred = []
        pos = 0
        NEG_INF = -1e9
        for local_idx, graph_idx in enumerate(batch_idxes):
            idx_map = idx_map_list[local_idx]
            num_nodes = len(idx_map)
            cur_pred = np.full(num_nodes, NEG_INF, dtype=np.float32)

            for node_idx in range(num_nodes):
                if idx_map[node_idx] < 0:
                    continue
                cur_pred[node_idx] = raw[pos]
                pos += 1

            # Mask covered nodes
            covered_nodes = covered[graph_idx] if covered[graph_idx] is not None else []
            for node in covered_nodes:
                if 0 <= node < num_nodes:
                    cur_pred[node] = NEG_INF

            # Mask infected nodes
            infect_list = g_list[graph_idx].graph.get('infect_list')
            if infect_list is not None:
                for node_idx, flag in enumerate(infect_list):
                    if flag == 1 and node_idx < num_nodes:
                        cur_pred[node_idx] = NEG_INF

            # Restrict to the risk-reachable set if it still has a valid candidate
            covered_set = set(covered_nodes)
            risk_set = PrepareBatchGraph.compute_risk_reachable(g_list[graph_idx], covered_set)
            has_valid_risk = False
            for node_idx in range(num_nodes):
                if cur_pred[node_idx] > NEG_INF and node_idx in risk_set:
                    has_valid_risk = True
                    break
            if has_valid_risk:
                for node_idx in range(num_nodes):
                    if node_idx not in risk_set:
                        cur_pred[node_idx] = NEG_INF

            pred.append(cur_pred)

        assert pos == len(raw)
        return pred

    def PredictWithCurrentQNet(self, g_list, covered):
        return self.Do_Predict(g_list, covered, False)

    def Do_PredictWithSnapshot(self, g_list, covered):
        return self.Do_Predict(g_list, covered, True)

    def TakeSnapShot(self):
        self.session.run(self.UpdateTargetQNetwork)

    # ======================== Training ========================

    def Fit(self):
        """One n-step Double DQN update."""
        sample = self.nStepReplayMem.sampling(BATCH_SIZE)
        ness = False
        for i in range(BATCH_SIZE):
            if not sample.list_term[i]:
                ness = True
                break
        if ness:
            double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
            double_list_predT = self.Do_PredictWithSnapshot(sample.g_list, sample.list_s_primes)
            list_pred = [a[my_argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]

        list_target = np.zeros([BATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            q_rhs = 0
            if not sample.list_term[i]:
                q_rhs = GAMMA * list_pred[i]
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs

        return self.do_fit(sample.g_list, sample.list_st, sample.list_at, list_target)

    def do_fit(self, g_list, covered, actions, list_target):
        loss = 0.0
        n_graphs = len(g_list)
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = min(BATCH_SIZE, n_graphs - i)
            batch_idxes = np.arange(i, i + bsize, dtype=np.int32)

            inputs = PrepareBatchGraph().setup_train(batch_idxes, g_list, covered, actions)
            my_dict = {
                self.rep_global: inputs['rep_global'],
                self.aux_input: np.array(inputs['aux_input']),
                self.target: list_target,
                self.action_select_risk: inputs['action_select_risk'],
                self.rep_global_risk: inputs['rep_global_risk'],
                self.n2nsum_param_risk: inputs['n2nsum_param_risk'],
                self.laplacian_param_risk: inputs['laplacian_param_risk'],
                self.subgsum_param_risk: inputs['subgsum_param_risk'],
                self.node_feat_risk: np.array(inputs['node_feat_risk'], dtype=np.float32).reshape((-1, 2)),
                self.graph_feat_risk: np.array(inputs['graph_feat_risk'], dtype=np.float32).reshape((-1, 2)),
                self.risk_align: inputs['risk_align'],
            }
            result = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)
            loss += result[0] * bsize
        return loss / len(g_list)

    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)

        for i in range(10):
            self.Run_simulator(2, 1, TrainSet=self.TrainSet)
        self.TakeSnapShot()
        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0

        save_dir = os.path.join('.', 'models', 'train')
        os.makedirs(save_dir, exist_ok=True)
        f_out = open(os.path.join(save_dir, 'valid_curve.csv'), 'w')
        f_out.write('# HDA:%.16f HBA:%.16f\n' % (self.baseline_hda, self.baseline_hba))
        for iter in range(MAX_ITERATION):
            start = time.time()
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            if iter % 10 == 0:
                self.Run_simulator(2, eps, TrainSet=self.TrainSet)
            if iter % 300 == 0:
                if iter == 0:
                    N_start = start
                else:
                    N_start = N_end

                frac = 0.0
                test_start = time.time()
                for idx in tqdm(range(n_valid), desc='Validating'):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%.16f\n' % (frac / n_valid))
                f_out.flush()
                print('iter %d, eps %.4f, average robustness:%.6f' % (iter, eps, frac / n_valid))
                print('testing %d graphs time: %.2fs' % (n_valid, test_end - test_start))
                N_end = time.time()
                print('300 iterations total time: %.2fs\n' % (N_end - N_start))
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            self.Fit()
        f_out.close()

    def Test(self, gid):
        """Robustness of the greedy removal sequence on a validation graph."""
        g_list = []
        self.test_env.s0_init_with_graph(self.TestSet.get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        while not self.test_env.is_terminal():
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = my_argMax(list_pred[0])
            self.test_env.step_without_reward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].number_of_nodes()))
        solution = sol + list(set(nodes) ^ set(sol))
        Robustness, _ = graph_utils.get_robustness(g_list[0], solution)
        return Robustness

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    # ======================== Evaluation ========================

    @staticmethod
    def LoadGraph(data_test):
        """Load an edge list and its infected-node labels (<name>_infected5.txt)."""
        g = nx.read_edgelist(data_test)
        g = nx.relabel_nodes(g, {node: int(node) for node in g.nodes()})
        infect_fn = os.path.splitext(data_test)[0] + '_infected5.txt'
        if not os.path.exists(infect_fn):
            raise FileNotFoundError('No infected file: %s' % infect_fn)
        with open(infect_fn) as f:
            infect_list = [int(item.strip()) for item in f.readlines()]
        g.graph['infect_list'] = np.array(infect_list, dtype=int)
        return g

    def EvaluateData(self, data_test, save_dir, stepRatio=0.0025):
        """Compute the removal sequence for a network and save it to save_dir/<name>.txt."""
        os.makedirs(save_dir, exist_ok=True)
        result_file = os.path.join(save_dir, os.path.basename(data_test))
        g = self.LoadGraph(data_test)
        if stepRatio > 0:
            step = max(int(stepRatio * nx.number_of_nodes(g)), 1)
        else:
            step = 1
        self.InsertGraph(g, is_test=True)
        t1 = time.time()
        solution = self.GetSolution(0, step)
        t2 = time.time()
        with open(result_file, 'w') as f_out:
            for node in solution:
                f_out.write(f'{node}\n')
        self.ClearTestGraphs()
        return solution, t2 - t1

    def GetSolution(self, gid, step=1):
        """Greedy removal sequence, removing the top-`step` nodes per forward pass."""
        g_list = []
        self.test_env.s0_init_with_graph(self.TestSet[gid])
        g_list.append(self.test_env.graph)
        sol = []
        while not self.test_env.is_terminal():
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if self.test_env.is_terminal():
                    break
                if new_action in self.test_env.covered_set:
                    continue
                infect_list = self.test_env.graph.graph.get('infect_list')
                if infect_list is not None and infect_list[new_action] != 0:
                    continue
                self.test_env.step_without_reward(new_action)
                sol.append(new_action)
        return sol

    def EvaluateSol(self, data_test, sol_file):
        """Robustness and per-step score curve of a saved removal sequence."""
        sys.stdout.flush()
        g = self.LoadGraph(data_test)
        print(data_test)
        print('number of nodes:%d' % nx.number_of_nodes(g))
        print('number of edges:%d' % nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print('number of sol nodes:%d' % len(sol))
        sol_left = list(set(nodes) ^ set(sol))
        solution = sol + sol_left
        Robustness, score_curve = graph_utils.get_robustness(g, solution)
        return Robustness, score_curve
