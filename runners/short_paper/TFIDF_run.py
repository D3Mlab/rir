# Mahdi Abdollahpour, 16/03/2022, 10:53 AM, PyCharm, Neural_PM

from Neural_PM.prefernce_matching import PM_experiment

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


setup = {
    'BERT_name': 'TFIDF',  # use keys of BERT_MODELS dictionary
    'create_matrices': False,  # if true always creats, if false checks if not exits creates
    'save_path': './short_paper/TFIDF/',  # choose a uniqe name between diffrent bert,split settings
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

setup['add_details_to_path'] = False
setup['tf_idf'] = True
e, o = PM_experiment.run_experiments(**setup)

directory = args.results_save_path
results_path = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith("metrics.txt"):
        results_path.append(f)


