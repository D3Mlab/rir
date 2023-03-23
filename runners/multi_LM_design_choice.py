# Created by mohammadmahdiabdollahpour at 2022-06-13


from Neural_PM.utils.exp import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching import PM_experiment
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS
from Neural_PM.finetune.train_utils import setup_tpu
import os
import argparse
from Neural_PM.utils.process_results import process, merge_dfs
from Neural_PM.utils.eval import mean_confidence_interval

# setup_list = get_debug_settings(1,[1,3])



strategy = setup_tpu()


LM_names = []
dfs = []
for lm in LM_names:
    stripped_BN = lm.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    BERT_MODELS[stripped_BN] = lm
    TOEKNIZER_MODELS[stripped_BN] = 'bert-base-uncased'
    setup_list = get_settings(cosine=False, bert_names=[stripped_BN], split_sentence=[False, True],
                              filtered_review_data='../data/50_above3.csv', true_labels_path='../data/new_binary.csv',
                              results_save_path='../results/generated_result_' + lm[-10:])
    setup_list = add_cosine(setup_list)
    for i, setup in enumerate(setup_list):
        print('Exp #', (i + 1))
        setup['strategy'] = strategy
        setup['from_pt'] = False
        setup['prepend_categories'] = False
        e, o = PM_experiment.run_experiments(**setup)

    directory = '../results/generated_result_' + lm[-10:]
    results_path = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith("metrics.txt"):
            results_path.append(f)

    #
    results_pd = process(results_path)
    results_pd = results_pd.sort_values(
        by=['similarity', 'BERT prefernce_matching', 'Review aggregation', 'k_R', 'Sentence aggregation', 'k_S', 'size',
            'subsample', 'seed'],
        axis=1)
    dfs.append(results_pd)
    results_pd.to_csv('../results' + lm[-10:] + '.csv', index=False)

df = merge_dfs(dfs)
df.to_csv('../dc.csv')

# print(results_pd.to_markdown())
