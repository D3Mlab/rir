from Neural_PM.prefernce_matching.statics import *
from os.path import join


from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.prefernce_matching.LM import *
import json
import numpy as np

import numpy as np

class ItemPreferenceMatching:
    def __init__(self, restaurants, id_name_map, id_to_index_path, item_embeddings_path, save_directory,
                 cosine=False, seed=None):
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
        q_matrix, q_mapping = load_query_matrices(save_directory, normalize=cosine)

        print("Embeddings Loaded from", item_embeddings_path)
        self.item_embeddings = np.load(item_embeddings_path)

        self.restaurants = restaurants
        self.q_matrix = q_matrix
        # #Q X representation size (768)

        # if no sentence agg { Restaurant : M #rev X 768}
        # if  sentence agg { Restaurant : M #sentences X 768}

        # {business_ID :{Text review: id of index in matrices}}
        self.q_mapping = q_mapping
        # {query text: id of query in q_matrix}
        self.cosine = cosine
        # boolean
        self.id_name_map = id_name_map

        # {business_id: Restaurant name}
        if id_to_index_path is not None:
            with open(id_to_index_path) as f:
                data = f.read()
            self.id_to_index = json.loads(data.replace('\'', '\"'))
        else:
            inv_id_name_map = {v: k for k, v in self.id_name_map.items()}
            self.id_to_index = {}
            for i, key in enumerate(list(inv_id_name_map.keys())):
                self.id_to_index[key] = i

    def predict(self, ):
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
            result = self.item_embeddings[self.id_to_index[bus_id]].dot(self.q_matrix.T)
            score = result.T

            output.append(score.reshape(-1, 1))

        outputs = np.concatenate(output, axis=1)

        return outputs, ''