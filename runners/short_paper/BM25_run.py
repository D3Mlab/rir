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
setup = {
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


if args.save_path is not None:
    setup['save_path'] = args.save_path
setup['add_details_to_path'] = False
setup['prepend_categories'] = args.prepend_categories
setup['BM25'] = True
e, o = BM25_experiment.run_experiments(**setup)
