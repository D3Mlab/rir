
from Neural_PM.utils.exp import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching.PM_experiment import BM25_experiment
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS
from Neural_PM.finetune.train_utils import setup_tpu
import os
import argparse

parser = argparse.ArgumentParser(description='Run Neural Preference Matching')

parser.add_argument('--prepend_categories', action='store_true')
parser.add_argument('--true_labels_path', type=str,
                    default='./data/PMD.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
parser.add_argument('--results_save_path', type=str,
                    default='./results/generated_result')
parser.add_argument('--save_path', type=str)
args = parser.parse_args()
setup_list = get_settings(cosine=False, bert_names='BM25', split_sentence=[False],
                          filtered_review_data=args.filtered_review_data, true_labels_path=args.true_labels_path,
                          results_save_path=args.results_save_path)
for i, setup in enumerate(setup_list):

    print('Exp #', (i + 1))
    if args.save_path is not None:
        setup['save_path'] = args.save_path

    setup['prepend_categories'] = args.prepend_categories
    setup['BM25'] = True
    e, o = BM25_experiment.run_experiments(**setup)
