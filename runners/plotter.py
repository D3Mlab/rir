# Mahdi Abdollahpour, 17/03/2022, 10:02 AM, PyCharm, lgeconvrec


from Neural_PM.utils import plotter
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Run Neural Preference Matching')

parser.add_argument('--result_path', type=str,
                    default='results.csv')

parser.add_argument('--LM_names', type=str,
                    default='TASB,VANILLA')
args = parser.parse_args()
df = pd.read_csv(args.result_path)
# bert_name = 'MSMARCO'
# sim = 'dot'
# bns = ['TASB', 'VANILLA']
bns = LM_names.split(',')
sims = ['dot', 'cosine']

for bert_name in bns:
    for sim in sims:
        plotter.plot_results(bert_name, sim, df)
