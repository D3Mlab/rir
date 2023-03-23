# Mahdi Abdollahpour, 16/03/2022, 10:53 AM, PyCharm, Neural_PM


# df = load_toronto_dataset(sample=1)
# df = filter_reviews(df, [filter_by_num_reviews(thres=400, compare="lte"), filter_by_rating(thres=3, compare="lt")])
# len(df)
from Neural_PM.utils.exp import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching import PM_experiment
from Neural_PM.prefernce_matching.PM_experiment import BM25_experiment
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS
import os
import argparse

parser = argparse.ArgumentParser(description='Run Neural Preference Matching')

parser.add_argument('--true_labels_path', type=str,
                    default='./data/PMD.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
parser.add_argument('--results_save_path', type=str,
                    default='./results/generated_result')

args = parser.parse_args()
models = ['TF-IDF']
models_keys = []
for model_name in models:
    name = model_name.split('/')[-1]
    stripped_BN = name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    models_keys.append(stripped_BN)
    BERT_MODELS[stripped_BN] = model_name
    TOEKNIZER_MODELS[stripped_BN] = model_name
# setup_list = get_debug_settings(1,[1,3])
from Neural_PM.finetune.train_utils import setup_tpu


strategy = None

setup_list = get_short_paper_settings(bert_names=models_keys, filtered_review_data=args.filtered_review_data,
                                      true_labels_path=args.true_labels_path,
                                      results_save_path=args.results_save_path)

size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350]

new_setup_list = []
for setup in setup_list:
    for size in size_list:
        setup['rate_size'] = size
        setup['subsample'] = True
        new_setup_list.append(setup.copy())

for i, setup in enumerate(new_setup_list):
    print('Exp #', (i + 1))
    setup['strategy'] = strategy
    setup['from_pt'] = True
    setup['add_details_to_path'] = True
    if setup['BERT_name'] == 'TF#IDF' or setup['BERT_name'] == 'TF-IDF':
        print('tf idf experiment')
        setup['tf_idf'] = True
        setup['strategy'] = None
        setup['from_pt'] = False
    e, o = PM_experiment.run_experiments(**setup)




BM25_setup = {
    'BERT_name': 'BM25',  # use keys of BERT_MODELS dictionary

    'create_matrices': False,  # if true always creats, if false checks if not exits creates
    'save_path': './short_paper/BM25/',  # choose a uniqe name between diffrent bert,split settings
    'agg_method': 'topk',  # one of 'topk','avg','max'
    'k': 1,  # for review aggregation
    'split_sentence': False,
    'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
    'sentence_k': 1,  # only used when split_sentence=True
    'save_output': True,  # save to file
    'save_metrics': True,  # save to file
    'rate_size': 1,
    'results_save_path': args.results_save_path,
    'cosine': False,
    'true_labels_path': args.true_labels_path,
    'filtered_review_data': args.filtered_review_data,
    'strategy': None,
}


size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350]

BM25_setup_list = []

for size in size_list:
    BM25_setup['rate_size'] = size
    BM25_setup['subsample'] = True
    BM25_setup_list.append(BM25_setup.copy())
for setup in BM25_setup_list:
    # if args.save_path is not None:
    #     setup['save_path'] = args.save_path
    setup['add_details_to_path'] = True
    # setup['prepend_categories'] = args.prepend_categories
    setup['BM25'] = True
    e, o = BM25_experiment.run_experiments(**setup)



directory = args.results_save_path
results_path = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith("metrics.txt"):
        results_path.append(f)

from Neural_PM.utils.process_results import process, process_per_query

#
results_pd = process(results_path)
print(results_pd)
results_pd.to_csv('size_results_sparse.csv', index=False)
