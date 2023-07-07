# Mahdi Abdollahpour, 16/03/2022, 10:53 AM, PyCharm, Neural_PM


# df = load_toronto_dataset(sample=1)
# df = filter_reviews(df, [filter_by_num_reviews(thres=400, compare="lte"), filter_by_rating(thres=3, compare="lt")])
# len(df)
from Neural_PM.utils.exp import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching import PM_experiment
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS
from Neural_PM.finetune.train_utils import setup_tpu
import os
import argparse

parser = argparse.ArgumentParser(description='Run Neural Preference Matching')
parser.add_argument('--passage_max_seq_len', type=int, default=512)
parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--tokenizer_name', type=str, default="bert-base-uncased")
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--from_pt', action='store_true')
parser.add_argument('--split_sentence', action='store_true')
parser.add_argument('--prepend_categories', action='store_true')
parser.add_argument('--true_labels_path', type=str,
                    default='./data/PMD.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
parser.add_argument('--results_save_path', type=str,
                    default='./results/generated_result')

parser.add_argument('--save_path', type=str)

parser.add_argument('--item_embedding', action='store_true')
parser.add_argument('--hybrid', action='store_true')
parser.add_argument('--tf_idf', action='store_true')
parser.add_argument('--item_embeddings_path', type=str)
parser.add_argument('--id_to_index_path', type=str)

# TODO: pass matrices path if available
args = parser.parse_args()

stripped_BN = args.model_name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
BERT_MODELS[stripped_BN] = args.model_name
TOEKNIZER_MODELS[stripped_BN] = args.tokenizer_name
# setup_list = get_debug_settings(1,[1,3])


if args.tpu:
    strategy = setup_tpu()
else:
    strategy = None
if not (args.item_embedding or args.hybrid):
    setup_list = get_settings(cosine=args.cosine, bert_names=[stripped_BN], split_sentence=[args.split_sentence],
                              filtered_review_data=args.filtered_review_data, true_labels_path=args.true_labels_path,
                              results_save_path=args.results_save_path)
    for i, setup in enumerate(setup_list):

        print('Exp #', (i + 1))
        if args.save_path is not None:
            setup['save_path'] = args.save_path
        setup['strategy'] = strategy
        setup['from_pt'] = args.from_pt
        setup['prepend_categories'] = args.prepend_categories
        setup['tf_idf'] = args.tf_idf
        e, o = PM_experiment.run_experiments(**setup)
elif args.item_embedding:
    e, o = PM_experiment.run_experiments_with_item_embedding(args.item_embeddings_path, args.id_to_index_path,
                                                             args.filtered_review_data, BERT_name=stripped_BN,
                                                             true_labels_path=args.true_labels_path,
                                                             # TODO: add split sentence to save_path
                                                             save_path='matrices_' + stripped_BN,
                                                             results_save_path=args.results_save_path,
                                                             strategy=strategy, from_pt=args.from_pt)
else:
    print('Hybrid...')
    e, o = PM_experiment.run_experiments_hybrid(item_embeddings_path=args.item_embeddings_path,
                                                id_to_index_path=args.id_to_index_path,
                                                filtered_review_data=args.filtered_review_data, BERT_name=stripped_BN,
                                                k=1, create_matrices=False,
                                                agg_method='topk',
                                                split_sentence=False, save_path='matrices_' + stripped_BN,
                                                true_labels_path=args.true_labels_path,
                                                results_save_path=args.results_save_path,
                                                strategy=strategy, from_pt=args.from_pt)

directory = args.results_save_path
results_path = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith("metrics.txt"):
        results_path.append(f)

from Neural_PM.utils.process_results import process


results_pd = process(results_path)
results_pd = results_pd.sort_values( by=['BERT prefernce_matching'],
                                     axis=0)
print(results_pd.to_markdown())
results_pd.to_csv('results.csv', index=False)
