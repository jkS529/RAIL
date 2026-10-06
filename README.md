# RAIL: Risk-Aware Isolation Learning

Code for the paper *Target-Aware Network Dismantling with Limited Budgets*.

RAIL is a deep reinforcement learning framework for target-aware network dismantling: given a network and a set of known infected nodes, it produces a ranked sequence of uninfected nodes to remove so that as few nodes as possible remain connected to the infected nodes. At each step, RAIL extracts the residual subgraph connected to infected nodes, encodes it with a graph neural network, and scores the candidate nodes with an n-step Double DQN.

## Repository structure

```
RAIL/
├── code/
│   ├── RAIL.py                 # Model, training loop and evaluation
│   ├── env.py                  # Node-removal environment and reward
│   ├── prepare_batch_graph.py  # Builds sparse inputs for the risk-reachable subgraph
│   ├── replay_mem.py           # N-step experience replay
│   ├── disjoint_set.py         # Union-find tracking components with infected nodes
│   ├── graph_utils.py          # Evaluation of removal sequences
│   ├── train.py                # Training entry point
│   ├── test.py                 # Evaluation entry point
│   └── models/                 # Pretrained model
└── data/                       # Real-world networks and infected-node labels
```

## Requirements

Tested with Python 3.9, TensorFlow 2.6.0, NetworkX 3.2.1, NumPy 1.19.5, pandas 1.2.0, SciPy 1.10.1 and tqdm 4.66.5:

```bash
pip install -r requirements.txt
```

The code uses the TensorFlow 1.x graph API through `tensorflow.compat.v1`.

## Data

Each network in `data/` consists of two files:

- `<name>.txt`: undirected edge list, one edge `u v` per line, with nodes labeled `0 ... N-1`.
- `<name>_infected5.txt`: `N` lines; line `i` is `1` if node `i` is infected and `0` otherwise.

Crime, Digg, Enron, Epinions, Facebook, Gnutella31 and HI-II-14 are taken from the benchmark of Fan et al. (*Nature Machine Intelligence*, 2020); cbn, faa-preferred-routes-2010, Figeys_Human_Protein_2007 and vidal_human_protein_2005 are obtained from the Index of Complex Networks (ICON). For these networks, 5% of the nodes are randomly marked as infected. Project 90 is an HIV-risk contact network from Colorado Springs, where empirical risk proxies are used as infected nodes.

## Usage

All scripts are run from the `code/` directory.

### Evaluate the pretrained model

```bash
cd code
python test.py
```

`test.py` loads `models/nrange_30_50_iter_99900.ckpt`, computes the removal sequence for each dataset in `DATA_NAMES`, and evaluates it. To evaluate other networks, edit `DATA_NAMES` in `test.py`. At evaluation time, the top 1% of candidate nodes (by Q-value) are removed per forward pass (`stepRatio=0.01`).

Outputs in `results/test/`:

| File | Content |
|---|---|
| `<name>.txt` | Ranked removal sequence, one node per line |
| `<name>_curve.txt` | Score after each removal step |
| `score.csv` | Overall score and evaluation time per dataset |
| `time.csv` | Time to compute the removal sequence per dataset |

### Train from scratch

```bash
cd code
python train.py
```

The model is trained on Barabási–Albert graphs with 30–50 nodes, of which 20% are randomly marked as infected. Hyperparameters are defined at the top of `RAIL.py`. Checkpoints are saved to `models/train/` every 300 iterations, and the validation score on 200 synthetic graphs is written to `models/train/valid_curve.csv`.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
