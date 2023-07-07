from Neural_PM.utils.exp import *
from Neural_PM.utils.eval import *
from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.utils.load_data import *
from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching.LM import BERT_model
import os
import pandas as pd
import numpy as np
import pickle
from Neural_PM.finetune.train_utils import setup_tpu


def save_warmups(BERT_name, filtered_review_data, true_labels_path, save_path, item_warmups_save_name):
    df = pd.read_csv(filtered_review_data)
    restaurants = df.groupby(['name'])
    print(df.shape)

    true_labels = build_true_labels(true_labels_path)
    queries, restaurants = get_queries_and_resturants(true_labels_path)
    restaurants = restaurants.tolist()

    reviews, id_name_map = sort_reviews_by_business(df, restaurants)
    print(len(reviews))
    strategy = setup_tpu()
    if not os.path.exists(save_path):
        # BERT_model_name = BERT_MODELS[BERT_name]
        BERT_model_name = BERT_name
        # tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        tokenizer_name = BERT_name
        if strategy is not None:
            with strategy.scope():
                embedder = BERT_model(BERT_model_name, tokenizer_name, True)
        else:
            embedder = BERT_model(BERT_model_name, tokenizer_name, True)
        save_matrices(queries, reviews, save_path, embedder, split_sentence=False, strategy=strategy)

    name_to_id = {v: k for k, v in id_name_map.items()}
    keys = [name_to_id[res] for res in restaurants]
    q_matrix, q_mapping, matrices, mappings = load_matrices(save_path, keys, split_sentence=False,
                                                            normalize=False)

    item_warmups = {}
    for key in keys:
        item_warmups[key] = np.average(matrices[key], axis=0)
    matrix = np.zeros((50, 768))
    for i, key in enumerate(keys):
        matrix[1, :] = item_warmups[key].reshape(1, -1)

    with open(item_warmups_save_name + '_warmup_matrix.npy', 'wb') as f:
        np.save(f, matrix)
    with open(item_warmups_save_name + '.pickle', 'wb') as handle:
        pickle.dump(item_warmups, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_warmups(file_name, id_to_index):
    with open(file_name, "rb") as input_file:
        item_warmups = pickle.load(input_file)

    index_to_id = {v: k for k, v in id_to_index.items()}
    embs = []
    for i in range(50):
        embs.append(item_warmups[index_to_id[i]].reshape(1, -1))
    weight = np.concatenate(embs, axis=0)
    return weight
