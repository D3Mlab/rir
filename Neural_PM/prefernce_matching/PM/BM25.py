import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from Neural_PM.utils.exp import *
from Neural_PM.utils.eval import *
from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.utils.load_data import *
from Neural_PM.prefernce_matching.PM.item import *
from Neural_PM.prefernce_matching.PM.review import *
from Neural_PM.prefernce_matching.PM.hybrid import *
import os
import pandas as pd
from Neural_PM.clustering.preprocess import preprocess_text_df
from Neural_PM.prefernce_matching.PM_experiment.experiment_utils import *
from Neural_PM.clustering.vectorization import TFIDFVectorizerModel


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


class BM25PreferenceMatching:
    def __init__(self, filtered_review_data, true_labels_path, seed=None,
                 subsample=False, n=1):
        """

        :param restaurants: name of the restaurants
        :param id_name_map: business id to name dictionary
        :param save_directory: the directory of matrices to load
        :param split_sentence:  whether sentences were splitted when creating the matrices
        :param cosine: if true, takes cosine similarity, o.w. it take dot
        :param seed: random seed for subsampling
        :param subsample: if true, subsamples
        :param n: number of desired reviews in subsampling
        """

        df = pd.read_csv(filtered_review_data)

        df['index'] = df.index
        restaurants = df.groupby(['name'])
        print(df.shape)

        true_labels = build_true_labels(true_labels_path)
        queries, restaurants = get_queries_and_resturants(true_labels_path)
        restaurants = restaurants.tolist()
        queries = [preprocess_text_df(q) for q in queries]

        reviews, id_name_map = sort_reviews_by_business(df, restaurants)

        for key in reviews.keys():
            reviews[key] = [preprocess_text_df(r) for r in reviews[key]]

        # indexes, id_name_map = sort_reviews_by_business(df, restaurants, 'index')
        # print(len(reviews), len(indexes))
        self.reviews = reviews
        if subsample:
            for key in reviews.keys():
                revs = reviews[key]
                # id_to_text = {v: k for k, v in mapping.items()}
                random.seed(seed)
                l = list(range(len(revs)))
                random.shuffle(l)
                chosen = l[:n]
                new_revs = [revs[index] for index in chosen]
                reviews[key] = new_revs
        self.restaurants = restaurants

        # #Q X representation size (768)
        self.true_labels = true_labels
        # if no sentence agg { Restaurant : M #rev X 768}
        # if  sentence agg { Restaurant : M #sentences X 768}
        self.queries = queries
        # {business_ID :{Text review: id of index in matrices}}
        self.restaurants = restaurants
        # {query text: id of query in q_matrix}

        # boolean
        self.id_name_map = id_name_map
        # {business_id: Restaurant name}

    def predict(self, agg_method="topk", k=10):
        """

        :param agg_method: review aggregation method (topk,avg)
        :param k: k in topk review aggregation
        :param sentence_agg_method: sentence aggregation method (topk,avg)
        :param sentence_agg_k: k in topk sentence aggregation
        :return: outputs score matrix of (queries, restaurants), best_matching_reviews
        """

        output = []
        inv_id_name_map = {v: k for k, v in self.id_name_map.items()}

        best_matching_reviews = [{} for i in range(len(self.queries))]
        bm25 = BM25()
        for iter in range(len(self.restaurants)):
            key = self.restaurants[iter]
            bus_id = inv_id_name_map[key]
            bm25.fit(self.reviews[bus_id])
        for iter in range(len(self.restaurants)):

            key = self.restaurants[iter]
            bus_id = inv_id_name_map[key]

            rr = []
            for q in self.queries:
                r = bm25.transform(q, self.reviews[bus_id])
                rr.append(r.reshape(1, -1))
            result = np.concatenate(rr, axis=0)
            # print(result.shape)
            score, indices, scores = aggregate_result(result, agg_func=agg_method, k=k)

            rows, cols = indices.shape

            inv_mapping = self.reviews[bus_id]

            for i in range(rows):
                for j in range(cols):
                    if bus_id in best_matching_reviews[i].keys():
                        best_matching_reviews[i][bus_id].append((inv_mapping[indices[i, j]], result[i, indices[i, j]]))
                    else:
                        best_matching_reviews[i][bus_id] = [(inv_mapping[indices[i, j]], result[i, indices[i, j]])]
            output.append(score.reshape(-1, 1))

        outputs = np.concatenate(output, axis=1)

        return outputs, best_matching_reviews
