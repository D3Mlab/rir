# Mahdi Abdollahpour, 16/03/2022, 10:53 AM, PyCharm, Neural_PM


# df = load_toronto_dataset(sample=1)
# df = filter_reviews(df, [filter_by_num_reviews(thres=400, compare="lte"), filter_by_rating(thres=3, compare="lt")])
# len(df)
from Neural_PM.utils.exp import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching import PM_experiment
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS
import os
import argparse

parser = argparse.ArgumentParser(description='Run Neural Preference Matching')
parser.add_argument('--passage_max_seq_len', type=int, default=512)

parser.add_argument('--tpu', action='store_true')
parser.add_argument('--true_labels_path', type=str,
                    default='./data/PMD.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
parser.add_argument('--results_save_path', type=str,
                    default='./results/generated_result')

args = parser.parse_args()
models = ['Luyu/co-condenser-marco', 'Luyu/condenser', 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
          'bert-base-uncased', 'facebook/contriever-msmarco']
models_keys = []
for model_name in models:
    name = model_name.split('/')[-1]
    stripped_BN = name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    models_keys.append(stripped_BN)
    BERT_MODELS[stripped_BN] = model_name
    TOEKNIZER_MODELS[stripped_BN] = model_name
# setup_list = get_debug_settings(1,[1,3])
from Neural_PM.finetune.train_utils import setup_tpu

if args.tpu:
    strategy = setup_tpu()
else:
    strategy = None

setup_list = get_short_paper_settings(bert_names=models_keys, filtered_review_data=args.filtered_review_data,
                                      true_labels_path=args.true_labels_path,
                                      results_save_path=args.results_save_path)
for i, setup in enumerate(setup_list):
    print('Exp #', (i + 1))
    setup['strategy'] = strategy
    setup['from_pt'] = True
    setup['add_details_to_path'] = False
    e, o = PM_experiment.run_experiments(**setup)

directory = args.results_save_path
results_path = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith("metrics.txt"):
        results_path.append(f)

from Neural_PM.utils.process_results import process, process_per_query

#
results_pd = process_per_query(results_path)
print(results_pd)
results_pd.to_csv('results.csv', index=False)
