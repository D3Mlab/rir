# Mahdi Abdollahpour, 08/04/2022, 09:54 AM, PyCharm, lgeconvrec


from itertools import combinations
import pandas as pd
import time
import random
import json
from sklearn.utils import shuffle
from itertools import combinations
import numpy as np
import pickle

# from sklearn.feature_extraction.text import TfidfVectorizer
# from bs4 import BeautifulSoup
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from numpy import dot
# from numpy.linalg import norm
# from sklearn.metrics.pairwise import cosine_similarity

cats = {'Alchemy Coffee': 'Coffee & Tea',
        'Bang Bang Ice Cream and Bakery': 'Desserts',
        'Bannock': 'Canadian (New)',
        'Barque Smokehouse': 'Barbeque',
        'Beerbistro': 'Bistros',
        "Blaze Fast-Fire'd Pizza": 'Pizza',
        'Buk Chang Dong Soon Tofu': 'Korean',
        'Byblos': 'Mediterranean',
        'Cactus Club Cafe': 'Cocktail Bars',
        'Cluny Bistro & Boulangerie': 'French',
        'Ding Tai Fung': 'Dim Sum',
        'Dragon Boat Fusion Cuisine': 'Asian Fusion',
        'Dumpling House Restaurant': 'Chinese',
        'Eggspectation': 'Breakfast & Brunch',
        'Fat Ninja Bite': 'Japanese',
        'Fishman Lobster Clubhouse Restaurant': 'Chinese',
        'GB Hand-pulled Noodles': 'Chinese',
        'Gusto 101': 'Italian',
        'Hokkaido Ramen Santouka': 'Ramen',
        'Insomnia Restaurant & Lounge': 'Lounges',
        'Inspire Restaurant': 'Asian Fusion',
        'JaBistro': 'Japanese',
        'KINKA IZAKAYA BLOOR': 'Japanese',
        'KINTON RAMEN': 'Ramen',
        'Kaka All You Can Eat': 'Japanese',
        'Kekou Gelato House': 'Desserts',
        'Khao San Road': 'Thai',
        'Konjiki Ramen': 'Ramen',
        'La Carnita': 'Mexican',
        'Lady Marmalade': 'Breakfast & Brunch',
        'Lahore Tikka House': 'Halal',
        'Lee Restaurant': 'Asian Fusion',
        "Maha's": 'Egyptian',
        'Manpuku Japanese Eatery': 'Japanese',
        'Messini Authentic Gyros': 'Greek',
        'Miku': 'Japanese',
        "Mildred's Temple Kitchen": 'Breakfast & Brunch',
        'Mill Street Brew Pub': 'Brewpubs',
        'Momofuku Noodle Bar': 'Ramen',
        "Mother's Dumplings": 'Chinese',
        'Muncheez': 'Creperies',
        'Nomé Izakaya': 'Lounges',
        'Pai Northern Thai Kitchen': 'Thai',
        'Patria': 'Spanish',
        'Pearl Diver': 'Seafood',
        'Pho Hung': 'Vietnamese',
        'Pizzeria Libretto': 'Pizza',
        'Playa Cabana': 'Mexican',
        'Ramen Isshin': 'Ramen',
        'Real Sports Bar & Grill': 'Sports Bars'}


def get_splits(all_reviews, finetune_setting):
    reviews_len = len(all_reviews)
    fold_number = finetune_setting['number']
    k = finetune_setting['repeat']
    print('Fold', fold_number, 'of', k)
    if k == 1:
        k = 5
    val_size = int(reviews_len / k)
    val_reviews = all_reviews[val_size * fold_number:val_size * (fold_number + 1)]
    if fold_number == 0:
        train_reviews = all_reviews[val_size * (fold_number + 1):]
    elif fold_number == k - 1:
        train_reviews = all_reviews[:val_size * fold_number]
    else:

        train_reviews = pd.concat([all_reviews[:val_size * fold_number], all_reviews[val_size * (fold_number + 1):]],
                                  axis=0)
    return train_reviews, val_reviews


def get_train_val_dfs(address_of_data, finetune_setting, additional_df=None):
    all_reviews = pd.read_csv(address_of_data)
    if additional_df is not None:
        all_reviews = pd.concat([all_reviews, additional_df], axis=0)
    all_reviews['index'] = all_reviews.index
    # if finetune_setting['prepend_categories']:
    #     all_reviews['review_text'] = all_reviews['categories'] + ' ' + all_reviews['review_text']

    all_reviews['review_id'] = all_reviews['business_id'] + all_reviews['user_id']
    # if finetune_setting['same_cat']:
    # all_reviews['cat'] = all_reviews['name'].apply(lambda x: cats[x])
    if finetune_setting['above_3']:
        all_reviews = all_reviews[all_reviews['review_stars'] > 2.5]
    # restaurants = all_reviews.business_id.unique()
    # this one needs a static random state
    all_reviews = shuffle(all_reviews, random_state=100)
    train_reviews, val_reviews = get_splits(all_reviews, finetune_setting)
    return train_reviews, val_reviews, all_reviews


def get_train_val_dfs_neural_ir(address_of_data, finetune_setting, additional_df=None, type_sampling='RC',
                                percent_of_query=0.55, seed=100):
    all_reviews = pd.read_csv(address_of_data)
    if additional_df is not None:
        all_reviews = pd.concat([all_reviews, additional_df], axis=0)
    all_reviews['index'] = all_reviews.index
    all_reviews['review_id'] = all_reviews['business_id'] + all_reviews['user_id']
    all_reviews['cat'] = all_reviews['name'].apply(lambda x: cats[x])
    # if finetune_setting['prepend_categories']:
    #     all_reviews['review_text'] = all_reviews['categories'] + ' ' + all_reviews['review_text']
    # restaurants = all_reviews.business_id.unique()
    if type_sampling == 'RC':
        applied_df = all_reviews.apply(lambda row: get_rc_spans(row.review_text, percent_of_query, seed),
                                       axis='columns',
                                       result_type='expand')
    else:
        applied_df = all_reviews.apply(lambda row: get_ic_spans(row.review_text, percent_of_query, seed),
                                       axis='columns',
                                       result_type='expand')
    df_t = pd.concat([all_reviews, applied_df], axis='columns')
    all_reviews = df_t.rename(columns={0: 'query', 1: 'doc'})
    # this one needs a static random state
    all_reviews = shuffle(all_reviews, random_state=100)

    # train_size = reviews_len - val_size
    train_reviews, val_reviews = get_splits(all_reviews, finetune_setting)

    return train_reviews, val_reviews, all_reviews


def get_positive_samples_contrastive(train_reviews, val_reviews, restaurants):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''

    all_positive_samples_train = {}
    all_positive_samples_val = {}
    all_len = 0
    val_len = 0

    print('Number of Restaurants:', len(restaurants))

    for restaurant in restaurants:
        temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
            ['review_text', 'index', 'review_stars']]
        temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][['review_text', 'index', 'review_stars']]
        # print(temp_df.shape,restaurant)

        id_rating_tuple_train = [tuple(x) for x in temp_df_train.to_numpy()]
        id_rating_tuple_val = [tuple(x) for x in temp_df_val.to_numpy()]

        positive_samples_train = list(combinations(id_rating_tuple_train, 2))
        positive_samples_val = list(combinations(id_rating_tuple_val, 2))
        all_len += len(id_rating_tuple_train)
        val_len += len(id_rating_tuple_val)
        all_positive_samples_train[restaurant] = positive_samples_train
        all_positive_samples_val[restaurant] = positive_samples_val
    print('Number of all positive samples:', all_len)
    print('Number of all val positive samples:', val_len)
    return all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews


def get_positive_samples_contrastive_same_category(train_reviews, val_reviews, categories):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''

    all_positive_samples_train = {}
    all_positive_samples_val = {}
    all_len = 0
    val_len = 0
    print('Number of Categories:', len(categories))

    for category in categories:
        temp_df_train = train_reviews[train_reviews['cat'] == category][['review_text', 'index', 'review_stars']]
        temp_df_val = val_reviews[val_reviews['cat'] == category][['review_text', 'index', 'review_stars']]
        # print(temp_df.shape,restaurant)

        id_rating_tuple_train = [tuple(x) for x in temp_df_train.to_numpy()]
        id_rating_tuple_val = [tuple(x) for x in temp_df_val.to_numpy()]

        positive_samples_train = list(combinations(id_rating_tuple_train, 2))
        positive_samples_val = list(combinations(id_rating_tuple_val, 2))
        all_len += len(id_rating_tuple_train)
        all_positive_samples_train[category] = positive_samples_train
        all_positive_samples_val[category] = positive_samples_val
    print('Number of all positive samples:', all_len)
    return all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews


def get_positive_samples_contrastive_same_cluster(train_reviews, val_reviews, restaurants):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''
    all_positive_samples_train = {}
    all_positive_samples_val = {}
    all_len = 0
    val_len = 0
    print('Number of Restaurants:', len(restaurants))

    for restaurant in restaurants:
        temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
            ['review_text', 'index', 'cluster_label']]
        temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][['review_text', 'index', 'cluster_label']]
        # print(temp_df.shape,restaurant)

        id_rating_tuple_train = [tuple(x) for x in temp_df_train.to_numpy()]
        id_rating_tuple_val = [tuple(x) for x in temp_df_val.to_numpy()]

        positive_samples_train = list(combinations(id_rating_tuple_train, 2))
        positive_samples_val = list(combinations(id_rating_tuple_val, 2))
        all_len += len(id_rating_tuple_train)
        all_positive_samples_train[restaurant] = positive_samples_train
        all_positive_samples_val[restaurant] = positive_samples_val
    print('Number of all positive samples:', all_len)
    return all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews


# def get_positive_samples_contrastive_with_index(address_of_data, seed=100):
#     '''
#     Generate all the combinations of reviews per restaurant
#     return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
#     '''
#
#     all_reviews = pd.read_csv(address_of_data)
#
#     all_reviews['index'] = all_reviews.index
#     all_reviews['review_id'] = all_reviews['business_id'] + all_reviews['user_id']
#     # all_reviews['cat'] = all_reviews['name'].apply(lambda x: cats[x])
#     restaurants = all_reviews.business_id.unique()
#     all_positive_samples_train = {}
#     all_positive_samples_val = {}
#     all_len = 0
#     all_len_val = 0
#     all_reviews = shuffle(all_reviews, random_state=seed)
#     reviews_len = len(all_reviews)
#     train_size = int(reviews_len * 0.8)
#     train_reviews = all_reviews[:train_size]
#     val_reviews = all_reviews[train_size:]
#     print('Number of Restaurants:', len(restaurants))
#
#     for restaurant in restaurants:
#         temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
#             ['review_text', 'index', 'review_stars']]
#         temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][['review_text', 'index', 'review_stars']]
#         # print(temp_df.shape,restaurant)
#
#         id_rating_tuple_train = [tuple(x) for x in temp_df_train.to_numpy()]
#         id_rating_tuple_val = [tuple(x) for x in temp_df_val.to_numpy()]
#
#         positive_samples_train = list(combinations(id_rating_tuple_train, 2))
#         positive_samples_val = list(combinations(id_rating_tuple_val, 2))
#         all_len += len(id_rating_tuple_train)
#         all_len_val += len(id_rating_tuple_val)
#         all_positive_samples_train[restaurant] = positive_samples_train
#         all_positive_samples_val[restaurant] = positive_samples_val
#     print('Number of all positive samples:', all_len)
#     print('Number of all positive samples val:', all_len_val)
#     return all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews


def get_positive_samples_contrastive_with_high_tfidf(train_reviews, val_reviews, restaurants, tfidf_file, threshold):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''

    with open(tfidf_file, 'rb') as fp:
        pairs = pickle.load(fp)

    all_positive_samples_train = {}
    all_positive_samples_val = {}
    all_len = 0
    val_len = 0
    print('Number of Restaurants:', len(restaurants))

    # positive_samples_train = {}
    # id_rating_tuple_val = {}
    # for restaurant in restaurants:
    #     positive_samples_train[restaurant] = []
    #     id_rating_tuple_val[restaurant] = []
    for restaurant in restaurants:
        temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
            ['review_text', 'index', 'review_stars']]
        temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][['review_text', 'index', 'review_stars']]
        # print(temp_df.shape,restaurant)
        id_rating_tuple_train = [x[0] for x in temp_df_train.to_numpy()]
        id_rating_tuple_val = [x[0] for x in temp_df_val.to_numpy()]
        positive_samples_train = []
        positive_samples_val = []
        for pair in pairs[restaurant]:
            # TODO: what threshold?
            if pair[1] > threshold:
                text1 = pair[0][0][0]
                rate1 = pair[0][0][1]
                text2 = pair[0][1][0]
                rate2 = pair[0][1][1]
                data_tuple = ((text1, rate1), (text2, rate2))
                if text1 in id_rating_tuple_train and text2 in id_rating_tuple_train:
                    positive_samples_train.append(data_tuple)
                if text1 in id_rating_tuple_val and text2 in id_rating_tuple_val:
                    positive_samples_val.append(data_tuple)

        # positive_samples_train = list(combinations(id_rating_tuple_train, 2))
        # positive_samples_val = list(combinations(id_rating_tuple_val, 2))
        all_len += len(id_rating_tuple_train)
        all_positive_samples_train[restaurant] = positive_samples_train
        all_positive_samples_val[restaurant] = positive_samples_val
    print('Number of all positive samples:', all_len)
    return all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews


def same_field_filter(positive_samples, index):
    keys = list(positive_samples.keys())
    results = {k: [] for k in keys}
    for key in keys:
        for sample in positive_samples[key]:
            if sample[0][index] == sample[1][index]:
                results[key].append(sample)
    return results


def diff_field_filter(positive_samples, index):
    keys = list(positive_samples.keys())
    results = {k: [] for k in keys}
    for key in keys:
        for sample in positive_samples[key]:
            if sample[0][index] != sample[1][index]:
                results[key].append(sample)
    return results


def different_field_filter(positive_samples, index, thresh=3):
    keys = list(positive_samples.keys())
    results = {k: [] for k in keys}
    for key in keys:
        for sample in positive_samples[key]:
            if abs(sample[0][index] - sample[1][index]) >= thresh:
                results[key].append(sample)
    return results


def positive_sub_sampler(input_dictionary, sample_size, seed):
    random.seed(seed)
    sub_sampled_positive = {}
    restaurants = input_dictionary.keys()
    for restaurant in restaurants:
        assert len(input_dictionary[restaurant]) >= sample_size, 'not enough reviews to subsample ' + str(
            len(input_dictionary[restaurant]))
        sub_sampled_positive[restaurant] = random.choices(input_dictionary[restaurant], k=sample_size)
    return sub_sampled_positive


def positive_sub_sampler_on_similarity(input_dictionary, sims, sample_size, seed, reverse=False, start=0):
    # print('Loading Embeddings from ', embedded_path)
    # Xfile = open(embedded_path, 'rb')
    # X = pickle.load(Xfile)
    # Xfile.close()
    #
    # sims = np.dot(X, X.T)
    # print('sims shape', sims.shape)
    # del X
    random.seed(seed)
    sub_sampled_positive = {}
    restaurants = input_dictionary.keys()
    for restaurant in restaurants:
        assert len(input_dictionary[restaurant]) >= sample_size, 'not enough reviews to subsample ' + str(
            len(input_dictionary[restaurant]))
        new_list = []
        for pair in input_dictionary[restaurant]:
            new_list.append([pair, sims[pair[0][1], pair[1][1]]])
        new_list.sort(key=lambda x: x[1], reverse=reverse)
        offset = max(0, int(start * (len(new_list) - sample_size)))
        selection = [x[0] for x in new_list[offset:offset + sample_size]]
        # sub_sampled_positive[restaurant] = random.choices(input_dictionary[restaurant], k=sample_size)
        sub_sampled_positive[restaurant] = selection
    return sub_sampled_positive


def get_hard_negatives(dict_subsampled, reviews, sims, hn_num=1):
    t = time.time()
    new_dict = {}
    restaurants = dict_subsampled.keys()
    for restaurant in restaurants:
        negatives = reviews[reviews['business_id'] == restaurant]
        this_restaurant_pairs = []
        for pair in dict_subsampled[restaurant]:
            new_list = []
            for index, row in negatives.iterrows():
                new_list.append([row['review_text'], sims[pair[0][1], row['index']]])
            new_list.sort(key=lambda x: x[1], reverse=True)
            hard_negatives = [x[0] for x in new_list[:hn_num]]
            this_restaurant_pairs.append((pair[0], pair[1], hard_negatives))
        new_dict[restaurant] = this_restaurant_pairs
    print('Mining hard negatives took:', time.time() - t)
    return new_dict




def read_all_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = list(map(json.loads, data))

    data = data[0]

    df = pd.DataFrame(data)
    df.rename(columns={'stars': 'review_stars', 'text': 'review_text'},
              inplace=True)

    df = df[['business_id', 'user_id', 'review_stars', 'review_text', 'name', 'categories', 'date']]
    return df


def get_asym_reviews(all_data_path, target_data_path):
    df_all = pd.read_csv(all_data_path)
    df_target = pd.read_csv(target_data_path)
    df = asym_restaurant_selection(df_all, df_target, 0, -1, 0)
    return df.review_text.values.tolist()


def get_sym_reviews(all_data_path, target_data_path, number_of_restaurants, min_review=50):
    df_all = pd.read_csv(all_data_path)
    df_target = pd.read_csv(target_data_path)
    df = asym_restaurant_selection(df_all, df_target, 0, number_of_restaurants, min_review)
    return df


def asym_restaurant_selection(df_all, df_target, min_avg_rating_restaurant, number_of_restaurants, min_review):
    # Getting only non target restaurants
    target_restaurants = list(df_target.groupby(['business_id']).count().index)
    df_all = df_all[~df_all['business_id'].isin(target_restaurants)]
    print('After target restaurant filter ', df_all.groupby(['business_id']).count().shape[0], ' left')
    # Filtering with minimum average rating
    average_review = df_all.groupby(['business_id']).mean()
    average_review_above = list(average_review[average_review['review_stars'] > min_avg_rating_restaurant].index)
    df_all = df_all[df_all['business_id'].isin(average_review_above)]
    print('After rating filters ', df_all.groupby(['business_id']).count().shape[0], ' left')
    # Filtering with minimum review count
    number_of_reviews = df_all.groupby(['business_id']).count()
    number_of_reviews = list(number_of_reviews[number_of_reviews['name'] > min_review].index)
    df_all = df_all[df_all['business_id'].isin(number_of_reviews)]
    # Filtering by number of restaurant in a sorted fashion
    exact_number_of_restaurants = list(df_all.groupby(['business_id']).count().sort_values(by=['business_id']).index)
    if len(exact_number_of_restaurants) > number_of_restaurants:
        df_all = df_all[df_all['business_id'].isin(exact_number_of_restaurants[:number_of_restaurants])]
    else:
        print('number of restaurants after filter ', len(exact_number_of_restaurants),
              ' is less than the desired number of restaurants ', number_of_restaurants)
    print('After all filters ', df_all.groupby(['name']).count().shape[0], ' left')
    return df_all


def get_rc_spans(text, percentage, seed):
    random.seed(seed)
    text_list = text.split(' ')
    all_length = len(text_list)
    span_length = int(all_length * percentage)
    i1 = random.randint(0, all_length - span_length - 1)
    i2 = random.randint(0, all_length - span_length - 1)
    return ' '.join(text_list[i1:i1 + span_length]), ' '.join(text_list[i2:i2 + span_length])


def get_ic_spans(text, percentage, seed):
    random.seed(seed)
    text_list = text.split(' ')
    all_length = len(text_list)
    span_length = int(all_length * percentage)
    i1 = random.randint(0, all_length - span_length - 1)

    return ' '.join(text_list[i1:i1 + span_length]), ' '.join(text_list[0:i1] + text_list[i1 + span_length + 1:])


def get_positive_samples_neural_ir(train_reviews, val_reviews, restaurants):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''
    all_positive_samples_train = {}
    all_positive_samples_val = {}
    all_len = 0
    print('Number of Restaurants:', len(restaurants))
    for restaurant in restaurants:
        temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
            ['review_text', 'query', 'doc', 'review_stars']]
        temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][
            ['review_text', 'query', 'doc', 'review_stars']]

        id_rating_tuple_train = [((x[1], x[3]), (x[2], x[3])) for x in temp_df_train.to_numpy()]
        id_rating_tuple_val = [((x[1], x[3]), (x[2], x[3])) for x in temp_df_val.to_numpy()]
        positive_samples_train = id_rating_tuple_train
        positive_samples_val = id_rating_tuple_val
        all_len += len(id_rating_tuple_train)
        all_positive_samples_train[restaurant] = positive_samples_train
        all_positive_samples_val[restaurant] = positive_samples_val
    print('Number of all positive samples:', all_len)
    return all_positive_samples_train, all_positive_samples_val


def get_positive_samples_item_embedding(train_reviews, val_reviews, restaurants):
    '''
    Generate all the combinations of reviews per restaurant
    return: dataframe with original data and review ID, return combination of restaurant in a dict={restaurant:[[(review1,rating1),(review2,rating2)],..[..],[..]]}
    '''

    all_positive_samples_train = {}
    all_positive_samples_val = {}
    t_len = 0
    v_len = 0

    print('Number of Restaurants:', len(restaurants))
    for restaurant in restaurants:
        temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][['review_text', 'review_stars']]
        temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][['review_text', 'review_stars']]

        all_positive_samples_train[restaurant] = list(temp_df_train.review_text.values)
        all_positive_samples_val[restaurant] = list(temp_df_val.review_text.values)
        t_len += len(all_positive_samples_train[restaurant])
        v_len += len(all_positive_samples_val[restaurant])
    print('Number of train positive samples for Item Embedding learning:', t_len)
    print('Number of val positive samples: for Item Embedding learning', v_len)
    return all_positive_samples_train, all_positive_samples_val


def get_positive_samples_using_upper_bound(train_reviews, val_reviews, restaurants, path_of_labels, negative_num,
                                           rev_from_res=100,
                                           seed=100, val_ratio=0.2):
    random.seed(seed)
    df = pd.read_csv(path_of_labels)
    queries = df["query"].unique()
    # restaurants = df["Restaurant name"].unique()
    relevant_items = {}
    for query in queries:
        relevant_items[query] = []
    for index, row in df.iterrows():
        if row["If only Low or  High"] == 1 and row["Restaurant name"] not in relevant_items[row["query"]]:
            relevant_items[row["query"]].append(row["Restaurant name"])
    data = []
    val_data = []
    for query in queries:
        negative_reviews_train = []
        negative_reviews_val = []
        for restaurant in restaurants:

            temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
                ['review_text', 'index', 'name']]
            temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][
                ['review_text', 'index', 'name']]
            name = temp_df_train.name.unique()[0]
            if name not in relevant_items[query]:
                train_text = temp_df_train.review_text.values
                val_text = temp_df_val.review_text.values
                negative_reviews_train.extend(train_text)
                negative_reviews_val.extend(val_text)

        for restaurant in restaurants:

            temp_df_train = train_reviews[train_reviews['business_id'] == restaurant][
                ['review_text', 'index', 'name']]
            temp_df_val = val_reviews[val_reviews['business_id'] == restaurant][
                ['review_text', 'index', 'name']]
            name = temp_df_train.name.unique()[0]
            if name in relevant_items[query]:
                train_text = random.sample(list(temp_df_train.review_text.values), rev_from_res)
                val_text = random.sample(list(temp_df_val.review_text.values), int(rev_from_res * val_ratio))
                for text in train_text:
                    data.append((query, text, random.sample(negative_reviews_train, negative_num)))
                for text in val_text:
                    val_data.append((query, text, random.sample(negative_reviews_val, negative_num)))
    return data, val_data


def read_review_dicts_upper_bound(data_list):
    standard_dicts = []
    for triple in data_list:
        # print(dict_subsampled[keys[j]][i])
        q = triple[0]
        rev2 = triple[1]
        sample = {}
        sample["query"] = q
        sample["passages"] = [{'text': rev2, 'label': 'positive'}]
        for hn in triple[2]:
            sample["passages"].append({'text': hn, 'label': 'hard_negative'})
        standard_dicts.append(sample)
    return standard_dicts
