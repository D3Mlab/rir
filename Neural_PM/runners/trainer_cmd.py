# Mahdi Abdollahpour, 09/04/2022, 01:48 PM, PyCharm, lgeconvrec


import argparse

parser = argparse.ArgumentParser(description='Train LM')
parser.add_argument('--passage_max_seq_len', type=int, default=512)
parser.add_argument('--query_max_seq_len', type=int, default=512)
parser.add_argument('--batch_size_per_replica', type=int, default=6)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--num_warmup_steps', type=int, default=1234)
parser.add_argument('--dropout', type=float, default=-1)
parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--ir_style', action='store_true')
parser.add_argument('--ir_sampling_type', type=str, default="RC")
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--reuse_matrices', action='store_true')
parser.add_argument('--high_tfidf', action='store_true')
parser.add_argument('--tfidf_scores_file', type=str, default="./data/tf_idfs.pickle")
parser.add_argument('--tfidf_threshold', type=float, default=0.8)
parser.add_argument('--same_rating', action='store_true')
parser.add_argument('--diff_rating', action='store_true')
parser.add_argument('--asym_negative', action='store_true')
parser.add_argument('--sym_negative', action='store_true')
parser.add_argument('--sym_negative_num', type=int, default=14)
# parser.add_argument('--asym_per_replica', type=int, default=1)
parser.add_argument('--run_neuralpm', action='store_true')
parser.add_argument('--train_data_path', type=str, default="./data/50_restaurants_all_rates.csv")
parser.add_argument('--all_data', type=str, default="./data/All_Toronto_reviews.csv")
parser.add_argument('--embedded_reviews_path', type=str, default="./data/BERT_embedded_reviews.pkl")
parser.add_argument('--positive_pair_per_restaurant', type=int, default=100)
parser.add_argument('--true_labels_path', type=str,
                    default='./data/PMD.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--repeat', type=int, default=1)

# TODO: add arguments
parser.add_argument('--item_embedding', action='store_true')
parser.add_argument('--freeze_lm', action='store_true')

parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--rate_diff', type=float, default=1.0)
parser.add_argument('--same_cat', action='store_true')
parser.add_argument('--same_cluster', action='store_true')
parser.add_argument('--diff_cluster', action='store_true')
parser.add_argument('--n_clusters', type=int, default=20)
parser.add_argument('--least_similar', action='store_true')
parser.add_argument('--most_similar', action='store_true')
parser.add_argument('--tfidf_feature', action='store_true')
parser.add_argument('--hard_negative', action='store_true')
parser.add_argument('--hard_negative_num', type=int, default=1)
parser.add_argument('--percentile', type=float, default=0.0)
parser.add_argument('--change_seed', action='store_true')
parser.add_argument('--prepend_categories', action='store_true')
parser.add_argument('--prepend_both', action='store_true')
parser.add_argument('--prepend_neuralpm', action='store_true')
parser.add_argument('--reverse_item_embedding', action='store_true')
parser.add_argument('--subsample_query', action='store_true')
parser.add_argument('--subsample_query_sentence', action='store_true')

parser.add_argument('--warmup', action='store_true')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--stddev', type=float, default=0.1)
parser.add_argument('--warmup_weights', type=str, default="item_embeddings_BERT.pickle")

args = parser.parse_args()

finetune_setting = {'passage_max_seq_len': args.passage_max_seq_len, 'query_max_seq_len': args.query_max_seq_len,
                    'batch_size_per_replica': args.batch_size_per_replica, "epochs": args.epochs,
                    "learning_rate": args.learning_rate, "num_warmup_steps": args.num_warmup_steps,
                    "dropout": args.dropout, "model_name": args.model_name, "tpu": args.tpu,
                    "train_data_path": args.train_data_path,
                    "positive_pair_per_restaurant": args.positive_pair_per_restaurant,
                    'true_labels_path': args.true_labels_path, 'filtered_review_data': args.filtered_review_data,
                    'patience': args.patience, 'gpu': args.gpu, 'run_neuralpm': args.run_neuralpm, 'number': 0,
                    'asym_negative': args.asym_negative, 'sym_negative': args.sym_negative,
                    # 'asym_per_replica': args.asym_per_replica,
                    'ir_style': args.ir_style, 'ir_sampling_type': args.ir_sampling_type, 'all_data': args.all_data,
                    'same_rating': args.same_rating, 'reuse_matrices': args.reuse_matrices,
                    'sym_negative_num': args.sym_negative_num,
                    # TODO: add item
                    'item_embedding': args.item_embedding,
                    'freeze_lm': args.freeze_lm,
                    'high_tfidf': args.high_tfidf,
                    'tfidf_scores_file': args.tfidf_scores_file,
                    'tfidf_threshold': args.tfidf_threshold,
                    'temperature': args.temperature,
                    'diff_rating': args.diff_rating,
                    'rate_diff': args.rate_diff,
                    'same_cat': args.same_cat,
                    'n_clusters': args.n_clusters,
                    'same_cluster': args.same_cluster,
                    'diff_cluster': args.diff_cluster,
                    'least_similar': args.least_similar,
                    'most_similar': args.most_similar,
                    'embedded_reviews_path': args.embedded_reviews_path,
                    'tfidf_feature': args.tfidf_feature,
                    'hard_negative': args.hard_negative,
                    'hard_negative_num': args.hard_negative_num,
                    'percentile': args.percentile,
                    'repeat': args.repeat,
                    'above_3': False,
                    'warmup': args.warmup,
                    'warmup_weights': args.warmup_weights,
                    'prepend_categories': args.prepend_categories,
                    'prepend_both': args.prepend_both,
                    'reverse_item_embedding': args.reverse_item_embedding,
                    'prepend_neuralpm': args.prepend_neuralpm,
                    'subsample_query': args.subsample_query,
                    'subsample_query_sentence': args.subsample_query_sentence,
                    'add_noise': args.add_noise,
                    'stddev': args.stddev,
                    }

from Neural_PM.finetune.train_experiment import run
from Neural_PM.utils.eval import mean_confidence_interval
import numpy as np
from Neural_PM.finetune.train import review_finetune, item_embedding_finetune

r_precs = []
maps = []
dfs = []
for i in range(args.repeat):

    finetune_setting['number'] = i
    if args.change_seed:
        finetune_setting['seed'] = (finetune_setting['number'] + 1) * 100
    else:
        finetune_setting['seed'] = 100
    print('Seed', finetune_setting['seed'])
    res_df = run(finetune_setting)
    dfs.append(res_df)
    # if args.run_neuralpm:
    #     print('R-Prec:', res_df['R-Prec'][0], 'MAP:', res_df['MAP'][0])
    #     r_precs.append(res_df['R-Prec'][0])
    #     maps.append(res_df['MAP'][0])
if args.run_neuralpm and args.repeat > 1:

    N = len(dfs[0])

    for i in range(N):
        by_q_type = {}
        r_precs = []
        maps = []
        for df in dfs:
            r_precs.append(df['R-Prec'][i])
            maps.append(df['MAP'][i])
            for key in df['by_query_type'][0].keys():
                if key not in by_q_type:
                    by_q_type[key] = [df['by_query_type'][i][key]]
                else:
                    by_q_type[key].append(df['by_query_type'][i][key])
        print('--->', df['Review aggregation'][i], df['k_R'][i])
        for key in by_q_type.keys():
            print(key)
            # print(r_precs)
            by_q_type_precs = [d['R-Prec'] for d in by_q_type[key]]

            ci = mean_confidence_interval(by_q_type_precs, 0.90)
            # print('mean:', np.mean(by_q_type_precs), 'CI:', ci)
            print('R-Prec', round(np.mean(by_q_type_precs), 4), '±', round(ci, 4))

            # print(key, 'MAP')
            # print(maps)
            by_q_type_maps = [d['MAP'] for d in by_q_type[key]]
            ci = mean_confidence_interval(by_q_type_maps, 0.90)
            # print('mean:', np.mean(by_q_type_maps), 'CI:', ci)
            print('MAP', round(np.mean(by_q_type_maps), 4), '±', round(ci, 4))
            print('-' * 7)

        print('R-Prec')
        print(r_precs)
        ci = mean_confidence_interval(r_precs, 0.90)
        print('mean:', np.mean(r_precs), 'CI:', ci)
        print('90 Conf-Int ', round(np.mean(r_precs), 4), '±', round(ci, 4))
        print('95 Conf-Int ', round(np.mean(r_precs), 4), '±', round(mean_confidence_interval(r_precs, 0.95), 4))

        print('MAP')
        print(maps)
        ci = mean_confidence_interval(maps, 0.90)
        print('mean:', np.mean(maps), 'CI:', ci)
        print('90 Conf-Int ', round(np.mean(maps), 4), '±', round(ci, 4))
        print('95 Conf-Int ', round(np.mean(maps), 4), '±', round(mean_confidence_interval(maps, 0.95), 4))
