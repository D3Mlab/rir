import argparse

parser = argparse.ArgumentParser(description='Cluster reviews')

parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--tfidf_feature', action='store_true')
parser.add_argument('--train_data_path', type=str, default="./data/50_restaurants_all_rates.csv")
parser.add_argument('--new_data_path', type=str, default="./data/data_with_cluster_labels.csv")
parser.add_argument('--embedded_reviews', type=str, default="./data/BERT_embedded_reviews.pkl")
parser.add_argument('--all_data', type=str, default="./data/All_Toronto_reviews.csv")
parser.add_argument('--positive_pair_per_restaurant', type=int, default=100)
parser.add_argument('--true_labels_path', type=str,
                    default='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/400_review_3_star_above - 400_review_3_star_above.csv')

parser.add_argument('--n_clusters', type=int, default=20)

args = parser.parse_args()

cluster_setting = {"model_name": args.model_name, "tpu": args.tpu,
                   "train_data_path": args.train_data_path,
                   'true_labels_path': args.true_labels_path, 'filtered_review_data': args.filtered_review_data,
                   'n_clusters': args.n_clusters,
                   'embedded_reviews': args.embedded_reviews,
                   'new_data_path': args.new_data_path,
                   'tfidf_feature': args.tfidf_feature,
                   }

from Neural_PM.clustering.clustering_experiment import run_clustering
from Neural_PM.utils.eval import mean_confidence_interval
import numpy as np

run_clustering(cluster_setting)
