# Mahdi Abdollahpour, 16/03/2022, 10:52 AM, PyCharm, Neural_PM


from nltk.util import pr
from os.path import join
from sklearn import preprocessing
import nltk

# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
from tqdm import tqdm
import numpy as np

import torch
import random
import time
import math


def save_matrices(queries, reviews, save_directory, embedder, split_sentence=False, strategy=None):
    """
    :param queries: list of queries text
    :param reviews: list of reviews text
    :param save_directory: to directory to save matrices
    :param embedder: the embedder function
    :param split_sentence: if true, splits the reviews into sentences

    this function does not return anything, it save the matrices in the save_directory

    :return:
    """
    start_time = time.time()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    N1 = len(queries)
    # print('N1',N1)
    # print(queries_df.head())
    q_matrix = np.zeros((N1, 768))
    q_mapping = {}
    idx = 0
    query_texts = list(queries)
    q_embed = embedder.apply_batch(query_texts, strategy)
    for i, text in enumerate(queries):
        # text=row["review_text"]

        # q_matrix[idx, :] = embedder.apply(text).cpu().detach().numpy()
        q_matrix[idx, :] = q_embed[i, :]
        if text in q_mapping.keys():
            text += '.'
        q_mapping[text] = idx
        idx += 1

    matrices = {}
    mappings = {}
    lens = {}

    for res_index, key in tqdm(enumerate(list(reviews.keys()))):
        # print('Indexing Rest.',res_index)
        if split_sentence:
            reviews_splited = []
            reviews_lens = []
            for r in reviews[key]:
                # rs = r.split('.')
                rs = sent_tokenize(r)
                reviews_splited.extend(rs)
                reviews_lens.append(len(rs))
            N2 = len(reviews_splited)
            review_list = reviews_splited
        else:
            N2 = len(reviews[key])
            review_list = reviews[key]

        r_matrix = np.zeros((N2, 768))
        r_mapping = {}
        idx = 0
        for text in reviews[key]:
            if text in r_mapping.keys():
                text += '.'
            r_mapping[text] = idx
            idx += 1
        CHUNK_SIZE = 800
        if strategy is not None and len(review_list) > CHUNK_SIZE:
            embeds = []
            chunks = math.ceil(len(review_list) / CHUNK_SIZE)
            for i in range(chunks):
                if i != chunks - 1:
                    rev_e = embedder.apply_batch(review_list[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], strategy)
                else:
                    rev_e = embedder.apply_batch(review_list[i * CHUNK_SIZE:], strategy)
                # print(rev_e.reshape(-1, 768).shape)
                rev_e = np.array(rev_e)
                embeds.append(rev_e.reshape(-1, 768))

            rev_embed = np.concatenate(embeds, axis=0)
            # print(rev_embed.shape)
        else:
            rev_embed = embedder.apply_batch(review_list, strategy)
        for i, text in enumerate(review_list):
            # r_matrix[idx, :] = embedder.apply(text).cpu().detach().numpy()
            r_matrix[i, :] = rev_embed[i, :]
            # idx += 1

        matrices[key] = r_matrix
        mappings[key] = r_mapping
        if split_sentence:
            lens[key] = reviews_lens
    # TODO: should not be torch tensor, save in numpy
    torch.save(q_matrix, join(save_directory, "q_matrix.pt"))
    torch.save(q_mapping, join(save_directory, "q_mapping.pt"))
    for key in matrices.keys():
        torch.save(mappings[key], join(save_directory, "r_mapping_" + key + ".pt"))
        torch.save(matrices[key], join(save_directory, "r_matrix_" + key + ".pt"))
        if split_sentence:
            torch.save(lens[key], join(save_directory, "r_lens_" + key + ".pt"))

    print("Creating Embeddings took", time.time() - start_time)


def save_matrices_tfidf(queries, reviews, save_directory, embedder, split_sentence=False, strategy=None):
    """
    :param queries: list of queries text
    :param reviews: list of reviews text
    :param save_directory: to directory to save matrices
    :param embedder: the embedder function
    :param split_sentence: if true, splits the reviews into sentences

    this function does not return anything, it save the matrices in the save_directory

    :return:
    """
    start_time = time.time()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    all_texts = []
    for res_index, key in tqdm(enumerate(list(reviews.keys()))):
        # print('Indexing Rest.',res_index)

        N2 = len(reviews[key])
        review_list = reviews[key]
        all_texts.extend(review_list)
    embedder.fit_transform(all_texts)




    N1 = len(queries)
    # print('N1',N1)
    # print(queries_df.head())
    nf = len(embedder.vectorizer.get_feature_names_out())
    print('nf',nf)
    q_matrix = np.zeros((N1, nf))
    q_mapping = {}
    idx = 0
    query_texts = list(queries)
    q_embed = embedder.apply_batch(query_texts, strategy)
    for i, text in enumerate(queries):
        # text=row["review_text"]

        # q_matrix[idx, :] = embedder.apply(text).cpu().detach().numpy()
        q_matrix[idx, :] = q_embed[i, :]
        if text in q_mapping.keys():
            text += '.'
        q_mapping[text] = idx
        idx += 1

    matrices = {}
    mappings = {}
    lens = {}

    for res_index, key in tqdm(enumerate(list(reviews.keys()))):
        # print('Indexing Rest.',res_index)
        if split_sentence:
            reviews_splited = []
            reviews_lens = []
            for r in reviews[key]:
                # rs = r.split('.')
                rs = sent_tokenize(r)
                reviews_splited.extend(rs)
                reviews_lens.append(len(rs))
            N2 = len(reviews_splited)
            review_list = reviews_splited
        else:
            N2 = len(reviews[key])
            review_list = reviews[key]

        r_matrix = np.zeros((N2, nf))
        r_mapping = {}
        idx = 0
        for text in reviews[key]:
            if text in r_mapping.keys():
                text += '.'
            r_mapping[text] = idx
            idx += 1
        CHUNK_SIZE = 800
        if strategy is not None and len(review_list) > CHUNK_SIZE:
            embeds = []
            chunks = math.ceil(len(review_list) / CHUNK_SIZE)
            for i in range(chunks):
                if i != chunks - 1:
                    rev_e = embedder.apply_batch(review_list[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], strategy)
                else:
                    rev_e = embedder.apply_batch(review_list[i * CHUNK_SIZE:], strategy)
                # print(rev_e.reshape(-1, 768).shape)
                rev_e = np.array(rev_e)
                embeds.append(rev_e.reshape(-1, nf))

            rev_embed = np.concatenate(embeds, axis=0)
            # print(rev_embed.shape)
        else:
            rev_embed = embedder.apply_batch(review_list, strategy)
        for i, text in enumerate(review_list):
            # r_matrix[idx, :] = embedder.apply(text).cpu().detach().numpy()
            r_matrix[i, :] = rev_embed[i, :]
            # idx += 1

        matrices[key] = r_matrix
        mappings[key] = r_mapping
        if split_sentence:
            lens[key] = reviews_lens
    # TODO: should not be torch tensor, save in numpy
    torch.save(q_matrix, join(save_directory, "q_matrix.pt"))
    torch.save(q_mapping, join(save_directory, "q_mapping.pt"))
    for key in matrices.keys():
        torch.save(mappings[key], join(save_directory, "r_mapping_" + key + ".pt"))
        torch.save(matrices[key], join(save_directory, "r_matrix_" + key + ".pt"))
        if split_sentence:
            torch.save(lens[key], join(save_directory, "r_lens_" + key + ".pt"))

    print("Creating Embeddings took", time.time() - start_time)

def save_query_matrices(queries, save_directory, embedder, strategy=None):
    """
    :param queries: list of queries text
    :param reviews: list of reviews text
    :param save_directory: to directory to save matrices
    :param embedder: the embedder function
    :param split_sentence: if true, splits the reviews into sentences

    this function does not return anything, it save the matrices in the save_directory

    :return:
    """
    start_time = time.time()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    N1 = len(queries)
    # print('N1',N1)
    # print(queries_df.head())
    q_matrix = np.zeros((N1, 768))
    q_mapping = {}
    idx = 0
    query_texts = list(queries)
    q_embed = embedder.apply_batch(query_texts, strategy)
    for i, text in enumerate(queries):
        # text=row["review_text"]

        # q_matrix[idx, :] = embedder.apply(text).cpu().detach().numpy()
        q_matrix[idx, :] = q_embed[i, :]
        if text in q_mapping.keys():
            text += '.'
        q_mapping[text] = idx
        idx += 1

    matrices = {}
    mappings = {}
    lens = {}

    # TODO: should not be torch tensor, save in numpy
    torch.save(q_matrix, join(save_directory, "q_matrix.pt"))
    torch.save(q_mapping, join(save_directory, "q_mapping.pt"))

    print("Creating Embeddings took", time.time() - start_time)


def load_matrices(save_directory, keys, split_sentence=False, normalize=False):
    """"
    :param
        save_directory: str,  path of th directory the matrices have been saved
        keys: [str] business ids of restaurant
        split_sentence: bool, if the sentences were splited during creating the matrices
        normalize: bool, if true normalized (for cosine distance)

    :return
    q_matrix: the matrix of query representation
    q_mapping: mapping from query text to index
    matrices: the matrix of query representation
    mappings:  mapping from query text to index
    lens: length
    """
    matrices = {}
    mappings = {}
    lens = {}

    q_matrix = torch.load(join(save_directory, "q_matrix.pt"))
    q_mapping = torch.load(join(save_directory, "q_mapping.pt"))
    if normalize:
        # print('normalizing')
        q_matrix = preprocessing.normalize(q_matrix, axis=1, norm='l2')

    # print(q_matrix.shape)
    for key in keys:
        mappings[key] = torch.load(join(save_directory, "r_mapping_" + key + ".pt"))
        matrices[key] = torch.load(join(save_directory, "r_matrix_" + key + ".pt"))
        if normalize:
            # print('normalizing')
            matrices[key] = preprocessing.normalize(matrices[key], axis=1, norm='l2')
        # print(matrices[key].shape)
        if split_sentence:
            lens[key] = torch.load(join(save_directory, "r_lens_" + key + ".pt"))

    if split_sentence:
        return q_matrix, q_mapping, matrices, mappings, lens
    else:
        return q_matrix, q_mapping, matrices, mappings


def load_query_matrices(save_directory, normalize=False):
    matrices = {}
    mappings = {}
    lens = {}

    q_matrix = torch.load(join(save_directory, "q_matrix.pt"))
    q_mapping = torch.load(join(save_directory, "q_mapping.pt"))
    if normalize:
        # print('normalizing')
        q_matrix = preprocessing.normalize(q_matrix, axis=1, norm='l2')
    return q_matrix, q_mapping


def aggregate_sentences(results, lens, agg_func='topk', k=5):
    """

    :param results: score matrix (queries,number of sentences of all review)
    :param lens: list of number of sentences of the reviews
    :param agg_func: aggregation function (topk,avg)
    :param k: k for topk aggregation
    :return: score matrix (queries,number reviews)
    """
    c_reviews_lens = [0]
    for l in lens:
        c_reviews_lens.append(c_reviews_lens[-1] + l)

    review_vectors = []
    for i in range(len(c_reviews_lens) - 1):
        # print(c_reviews_lens[i],c_reviews_lens[i+1],results.shape)
        chunk = results[:, c_reviews_lens[i]:c_reviews_lens[i + 1]]
        indices = np.argsort(chunk, axis=1)
        indices = np.flip(indices, axis=1)
        scores_full = np.take_along_axis(chunk, indices, axis=1)
        indices = indices[:, :k]
        if agg_func == 'topk':
            scores = scores_full[:, :k]
            score = np.average(scores, axis=1)
        elif agg_func == 'avg':
            score = np.average(chunk, axis=1)
        elif agg_func == 'max':
            score = np.max(chunk, axis=1)

        review_vectors.append(score.reshape(-1, 1))

    # print('review_vectors len',len(review_vectors))
    new_result = np.concatenate(review_vectors, axis=1)
    return new_result


def aggregate_result(results, agg_func='topk', k=5):
    """

    :param results: score matrix (queries,number reviews)
    :param agg_func: aggregation function (topk,avg)
    :param k: k for topk aggregation
    :return: score matrix (queries, restaurants)
    """
    indices = np.argsort(results, axis=1)
    indices = np.flip(indices, axis=1)
    scores_full = np.take_along_axis(results, indices, axis=1)
    if agg_func == 'topk':
        indices = indices[:, :k]

        scores = scores_full[:, :k]
        score = np.average(scores, axis=1)
    elif agg_func == 'avg':
        score = np.average(results, axis=1)
    elif agg_func == 'max':
        score = np.max(results, axis=1)
    elif agg_func == 'ratio':
        max_score = np.max(results, axis=1)
        thresh = max_score * k
        # print(max_score-thresh)
        # print(thresh)
        m, n = results.shape
        score = np.zeros_like(max_score)
        for i in range(m):
            index = 0
            for j in range(n):
                if scores_full[i, j] >= thresh[i]:
                    index += 1
                else:
                    break
            print(index)
            score[i] = np.average(scores_full[i, :index])
    return score, indices, scores_full


def subsampler(matrix, mapping, n, seed):
    """

    :param matrix: review embedding matrix
    :param mapping: restaurant name to business id mapping
    :param n: number of reviews wanted
    :param seed: the random seed
    :return:
    """
    # print('Subsampling...')

    m, t = matrix.shape
    id_to_text = {v: k for k, v in mapping.items()}
    random.seed(seed)
    l = list(id_to_text)
    random.shuffle(l)
    chosen = l[:n]
    new_mapping = {}
    new_matrix = np.zeros((n, t))
    for i, idx in enumerate(chosen):
        new_matrix[i, :] = matrix[idx, :]
        new_mapping[id_to_text[idx]] = i
    # print(new_matrix.shape,len(new_mapping),matrix.shape,len(mapping))
    return new_matrix, new_mapping
