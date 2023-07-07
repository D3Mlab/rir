import os
import json
from Neural_PM.utils.eval import *
from Neural_PM.utils.load_data import *
from pathlib import Path
import pandas as pd
import numpy as np


def per_query_result(results_path='/content/finetuning', save_path_results='/content/results_per_query.csv',
                     save_path_map='/content/results_per_query_map.csv', read_from_path_results=None,
                     read_from_path_map=None,
                     true_labels_path='/content/data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv'):
    last_index = 0
    results_address = []
    # If the is no file present
    if read_from_path_results == None:

        true_labels = build_true_labels(true_labels_path)
        queries, restaurants = get_queries_and_resturants(true_labels_path)
        df = pd.DataFrame(np.array(queries), columns=['queries'])
        df_map = pd.DataFrame(columns=['exp id', 'address'])

    # If previous results present
    else:
        df = pd.read_csv(read_from_path_results)
        df_map = pd.read_csv(read_from_path_map)
        last_index = int(df.columns[-1].split(':')[-1])

    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.endswith("metrics.txt"):
                results_address.append(os.path.join(root, file))

    for i, result_address in enumerate(results_address):
        f = open(result_address, 'r')
        fstring = f.read()
        fstring = str(fstring).replace("'", '"')
        jason_result = json.loads(fstring)
        results = jason_result['all_R_Prec']
        results = results.split(',')
        ind = i + last_index
        exp_id = 'Exp number:' + str(ind)
        df[exp_id] = results
        df2 = {'exp id': exp_id, 'address': results_address}
        df_map = df_map.append(df2, ignore_index=True)
    filepath = Path(save_path_results)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

    filepath_map = Path(save_path_map)
    filepath_map.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath_map)
