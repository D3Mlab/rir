from Neural_PM.prefernce_matching.statics import *
from os.path import join

from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.prefernce_matching.LM import *
import json
import numpy as np

import numpy as np


class HybridPreferenceMatching:
    def __init__(self,item_embeddings_path,id_to_index_path, restaurants, id_name_map, save_directory, split_sentence=False, cosine=False, seed=None,
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

        name_to_id = {v: k for k, v in id_name_map.items()}
        keys = [name_to_id[res] for res in restaurants]
        if split_sentence:
            q_matrix, q_mapping, matrices, mappings, lens = load_matrices(save_directory, keys,
                                                                          split_sentence=split_sentence,
                                                                          normalize=cosine)
            self.lens = lens
        else:
            q_matrix, q_mapping, matrices, mappings = load_matrices(save_directory, keys, split_sentence=split_sentence,
                                                                    normalize=cosine)
        print("Embeddings Loaded from", save_directory)
        if subsample:
            print('Subsampling...', seed, n)
            for key in matrices.keys():
                matrices[key], mappings[key] = subsampler(matrices[key], mappings[key], seed=seed, n=n)
        self.split_sentence = split_sentence
        self.restaurants = restaurants
        self.q_matrix = q_matrix
        # #Q X representation size (768)
        self.matrices = matrices
        # if no sentence agg { Restaurant : M #rev X 768}
        # if  sentence agg { Restaurant : M #sentences X 768}
        self.mappings = mappings
        # {business_ID :{Text review: id of index in matrices}}
        self.q_mapping = q_mapping
        # {query text: id of query in q_matrix}
        self.cosine = cosine
        # boolean
        self.id_name_map = id_name_map
        # {business_id: Restaurant name}
        print("Item Embeddings Loaded from", item_embeddings_path)
        self.item_embeddings = np.load(item_embeddings_path)

        with open(id_to_index_path) as f:
            data = f.read()
        self.id_to_index = json.loads(data.replace('\'', '\"'))

    def predict(self, agg_method="topk", k=10, sentence_agg_method='topk', sentence_agg_k=3):
        """

        :param agg_method: review aggregation method (topk,avg)
        :param k: k in topk review aggregation
        :param sentence_agg_method: sentence aggregation method (topk,avg)
        :param sentence_agg_k: k in topk sentence aggregation
        :return: outputs score matrix of (queries, restaurants), best_matching_reviews
        """

        output = []
        inv_id_name_map = {v: k for k, v in self.id_name_map.items()}

        best_matching_reviews = [{} for i in range(len(self.q_matrix))]
        for iter in range(len(self.restaurants)):

            key = self.restaurants[iter]
            bus_id = inv_id_name_map[key]
            item_result = self.item_embeddings[self.id_to_index[bus_id]].dot(self.q_matrix.T)
            item_score = item_result.T

            result = self.matrices[bus_id].dot(self.q_matrix.T)
            result = result.T
            if self.split_sentence:
                result = aggregate_sentences(result, self.lens[bus_id], agg_func=sentence_agg_method, k=sentence_agg_k)

            score, indices, scores = aggregate_result(result, agg_func=agg_method, k=k)

            rows, cols = indices.shape

            inv_mapping = {v: k for k, v in self.mappings[bus_id].items()}

            for i in range(rows):
                for j in range(cols):
                    if bus_id in best_matching_reviews[i].keys():
                        best_matching_reviews[i][bus_id].append((inv_mapping[indices[i, j]], result[i, indices[i, j]]))
                    else:
                        best_matching_reviews[i][bus_id] = [(inv_mapping[indices[i, j]], result[i, indices[i, j]])]

            score = score.reshape(-1, 1)
            score += item_score.reshape(-1, 1)

            output.append(score.reshape(-1, 1))

        outputs = np.concatenate(output, axis=1)

        return outputs, best_matching_reviews
