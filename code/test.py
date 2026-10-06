import os
import time

import numpy as np
import pandas as pd

from RAIL import RAIL

DATA_PATH = '../data/'
DATA_NAMES = ['Crime']
MODEL_FILE = 'models/nrange_30_50_iter_99900.ckpt'
SAVE_DIR = '../results/test/'


def GetSolution(stepRatio, model_file):
    """Compute removal sequences and save them to SAVE_DIR/<name>.txt and the running time to time.csv."""
    dqn = RAIL()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print('The best model is :%s' % model_file)
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.zeros((1, len(DATA_NAMES))), index=['time'], columns=DATA_NAMES)
    for j, name in enumerate(DATA_NAMES):
        print('\nTesting dataset %s' % name)
        data_test = DATA_PATH + name + '.txt'
        solution, solution_time = dqn.EvaluateData(data_test, SAVE_DIR, stepRatio)
        df.iloc[0, j] = solution_time
        print('Data:%s, time:%.2f' % (name, solution_time))
    df.to_csv(os.path.join(SAVE_DIR, 'time.csv'), encoding='utf-8', index=False)


def EvaluateSolution():
    """Evaluate saved removal sequences; save the score curve to <name>_curve.txt and scores to score.csv."""
    dqn = RAIL()
    df = pd.DataFrame(np.zeros((2, len(DATA_NAMES))), index=['solution', 'time'], columns=DATA_NAMES)
    for i, name in enumerate(DATA_NAMES):
        print('\nEvaluating dataset %s' % name)
        data_test = DATA_PATH + name + '.txt'
        solution = os.path.join(SAVE_DIR, name + '.txt')
        t1 = time.time()
        score, score_curve = dqn.EvaluateSol(data_test, solution)
        t2 = time.time()
        df.iloc[0, i] = score
        df.iloc[1, i] = t2 - t1
        with open(os.path.join(SAVE_DIR, name + '_curve.txt'), 'w') as f_out:
            for value in score_curve:
                f_out.write('%.8f\n' % value)
        print('Data: %s, score:%.6f' % (name, score))
    df.to_csv(os.path.join(SAVE_DIR, 'score.csv'), encoding='utf-8', index=False)


def main():
    GetSolution(0.01, MODEL_FILE)
    EvaluateSolution()


if __name__ == "__main__":
    main()
