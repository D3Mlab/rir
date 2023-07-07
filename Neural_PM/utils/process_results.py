# Mahdi Abdollahpour, 16/03/2022, 10:57 AM, PyCharm, Neural_PM


import json
import pandas as pd
import numpy as np
import os
from Neural_PM.utils.eval import mean_confidence_interval


def process(results_path):
    '''
    Read from the saved results
    :param results_path: the final path of metrics.txt s
    :return: A pandas data frame of experiment setting and results
    '''

    results = {}
    for i, result in enumerate(results_path):

        with open(result) as f:
            data = f.read()
        d = json.loads(data.replace('\'', '\"'))
        for key in d.keys():
            if key != 'all_R_Prec':
                if key in results.keys():
                    results[key].append(d[key])
                else:
                    assert i == 0, "missing data in at least one the result files"
                    results[key] = [d[key]]
    max_len = max([len(results[x]) for x in results.keys()])
    to_pd = {key: results[key] for key in results.keys() if len(results[key]) == max_len}
    results_pd = pd.DataFrame.from_dict(to_pd)
    return results_pd


def process_per_query(results_path):
    '''
    Read from the saved results
    :param results_path: the final path of metrics.txt s
    :return: A pandas data frame of experiment setting and results
    '''

    results = {}
    for i, result in enumerate(results_path):

        with open(result) as f:
            data = f.read()
        d = json.loads(data.replace('\'', '\"'))
        for key in d.keys():
            if key in results.keys():
                results[key].append(d[key])
            else:
                assert i == 0, "missing data in at least one the result files"
                results[key] = [d[key]]
    results_pd = pd.DataFrame.from_dict(results)
    return results_pd


def merge_dfs(dfs):
    N = len(dfs[0])
    new_data_frame = {}
    for key in dfs[0].columns:
        new_data_frame[key] = []
    for i in range(N):
        r_precs = []
        maps = []
        for df in dfs:
            r_precs.append(df['R-Prec'][i])
            maps.append(df['MAP'][i])
        for key in new_data_frame.keys():
            new_data_frame[key].append(dfs[0][key][i])
        ci = mean_confidence_interval(r_precs, 0.90)
        new_data_frame['R-Prec'][-1] = round(np.mean(r_precs), 4)
        new_data_frame['R-Prec_CI'][-1] = round(ci, 4)

        ci = mean_confidence_interval(maps, 0.90)
        new_data_frame['MAP'][-1] = round(np.mean(maps), 4)
        new_data_frame['MAP_CI'][-1] = round(ci, 4)
    df = pd.DataFrame(new_data_frame)
    return df
