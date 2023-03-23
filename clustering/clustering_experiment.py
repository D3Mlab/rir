from Neural_PM.clustering.clustering import *
from Neural_PM.finetune.train_utils import setup_tpu
import os
import pandas as pd

def run_clustering(clustering_setting):

    cr = ClusterReviews(clustering_setting["n_clusters"])
    all_reviews = pd.read_csv(clustering_setting["train_data_path"])
    texts = all_reviews.review_text.values.tolist()
    if os.path.exists(clustering_setting["embedded_reviews"]):
        print('Loading Embeddings')
        Xfile = open(clustering_setting["embedded_reviews"], 'rb')
        X = pickle.load(Xfile)
        Xfile.close()
    else:

        if clustering_setting['tfidf_feature']:
            print('Using TFIDF features for')
            vec_model = TFIDFVectorizerModel()
        else:
            if clustering_setting['tpu']:
                strategy = setup_tpu()
            else:
                strategy = None
            vec_model = BERTVectorizerModel(clustering_setting['model_name'], strategy)
        print('Embedding...')
        X = vec_model.get_features(texts)
        pickle.dump(X, open(clustering_setting["embedded_reviews"], 'wb'))  # Saving the model
    print('Clustering...')
    labels = cr.cluster(X)
    cr.save()
    all_reviews['cluster_label'] = labels
    all_reviews.to_csv(clustering_setting['new_data_path'])

