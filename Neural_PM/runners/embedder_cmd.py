import argparse

parser = argparse.ArgumentParser(description='Cluster reviews')

parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--tfidf_feature', action='store_true')
parser.add_argument('--train_data_path', type=str, default="./data/50_restaurants_all_rates.csv")
parser.add_argument('--embedded_reviews', type=str, default="./data/BERT_embedded_reviews.pkl")





args = parser.parse_args()

embedding_setting = {"model_name": args.model_name, "tpu": args.tpu,
                   "train_data_path": args.train_data_path,
                   'embedded_reviews': args.embedded_reviews,
                   'tfidf_feature': args.tfidf_feature,
                   }

from Neural_PM.clustering.clustering_experiment import run_clustering
from Neural_PM.utils.eval import mean_confidence_interval
import numpy as np
from Neural_PM.clustering.vectorization import *
from Neural_PM.finetune.train_utils import setup_tpu
import pandas as pd

all_reviews = pd.read_csv(embedding_setting["train_data_path"])
texts = all_reviews.review_text.values.tolist()
if embedding_setting['tfidf_feature']:
    print('Using TFIDF features')
    vec_model = TFIDFVectorizerModel()
else:
    if embedding_setting['tpu']:
        strategy = setup_tpu()
    else:
        strategy = None
    vec_model = BERTVectorizerModel(embedding_setting['model_name'], strategy)
print('Embedding...')
X = vec_model.get_features(texts)
pickle.dump(X, open(embedding_setting["embedded_reviews"], 'wb'))  # Saving the model
