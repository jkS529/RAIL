import random
from typing import List, Optional
import networkx as nx
from env import RiskEnv


class ReplaySample:
    def __init__(self, batch_size: int):
        self.g_list: List[nx.Graph] = []
        self.list_st: List[List[int]] = []
        self.list_s_primes: List[List[int]] = []
        self.list_at: List[int] = []
        self.list_rt: List[float] = []
        self.list_term: List[bool] = []


class NStepReplayMem:
    def __init__(self, memory_size: int, seed=None):
        self.memory_size = memory_size
        self.graphs: List[nx.Graph] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.states: List[List[int]] = []
        self.s_primes: List[List[int]] = []
        self.terminals: List[bool] = []

        self.current = 0
        self.count = 0
        self.generator = random.Random(seed)

    def add(self, g: nx.Graph, s_t: List[int], a_t: int, r_t: float, s_prime: List[int], terminal: bool):
        if self.count < self.memory_size:
            self.graphs.append(g)
            self.states.append(s_t)
            self.actions.append(a_t)
            self.rewards.append(r_t)
            self.s_primes.append(s_prime)
            self.terminals.append(terminal)
        else:
            self.graphs[self.current] = g
            self.states[self.current] = s_t
            self.actions[self.current] = a_t
            self.rewards[self.current] = r_t
            self.s_primes[self.current] = s_prime
            self.terminals[self.current] = terminal

        self.current = (self.current + 1) % self.memory_size
        self.count = min(self.count + 1, self.memory_size)

    def add_env(self, env: RiskEnv, n_step: int):
        """Store the n-step transitions of a finished episode."""
        assert env.is_terminal() or not env.has_available_risk_action()
        num_steps = len(env.state_seq)
        assert num_steps

        # sum_rewards[i] = sum of rewards from step i to the end of the episode
        env.sum_rewards[num_steps - 1] = env.reward_seq[num_steps - 1]
        for i in range(num_steps - 2, -1, -1):
            env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = env.action_list
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = env.state_seq[i + n_step]
                term_t = False

            self.add(env.graph, env.state_seq[i], env.act_seq[i], cur_r, s_prime, term_t)

    def sampling(self, batch_size: int) -> Optional[ReplaySample]:
        assert self.count >= batch_size

        indices = self.generator.sample(range(self.count), batch_size)

        result = ReplaySample(batch_size)
        for idx in indices:
            result.g_list.append(self.graphs[idx])
            result.list_st.append(self.states[idx])
            result.list_at.append(self.actions[idx])
            result.list_rt.append(self.rewards[idx])
            result.list_s_primes.append(self.s_primes[idx])
            result.list_term.append(self.terminals[idx])

        return result
