from itertools import combinations
import pandas as pd
import time
import random
import json
from sklearn.utils import shuffle
from itertools import combinations
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize

full_categories = {'DE89UdHFMCN6DtYWZuer5A': 'Ramen, Noodles', 'O-uIEuv7JLUHajkemx_sVw': 'Bistros, Beer Bar',
                   'xaqW4NkmUCGRRjmLxiFJ6Q': 'Creperies, Delis', 'N93EYZy9R0sdlEvubu94ig': 'Sandwiches',
                   'O1TvPrgkK2bUo5O5aSZ7lw': 'Dim Sum', '8YG7PRoZuHeKii0rtNvJFQ': 'Asian Fusion, Dim Sum',
                   'Qmwfg-PtYuCo5Q_IwcA_HQ': 'Japanese, Sushi Bars', 'BUcTdN-rNE8urCCQuxSOQA': 'Chinese',
                   'hPXR-Bi8U-uz6TUViqmGpg': 'Japanese, Burgers, Korean', 'iGEvDk6hsizigmXhDKs2Vg': 'Mexican',
                   'UxWH8zRYIBgs6Q2oykvRdw': 'French, Wine Bars, Cocktail Bars',
                   'KVpvE6pOPM9KMvak4HFsNg': 'Pizza, Italian',
                   'Yl2TN9c23ZGLUBSD9ks5Uw': 'Mediterranean, Middle Eastern',
                   'e41TP5cXZqSrz50xCBJqZw': 'Lounges, Breakfast & Brunch, Comfort Food',
                   '0a2O150ytxrDjDzXNfRWkA': 'Japanese, Sushi Bars',
                   'cQK9M2JAwETQnnBoYyua5A': 'Lounges, Breakfast & Brunch, Canadian (New)',
                   'oWTn2IzrprsRkPfULtjZtQ': 'Burgers', 'XYIPXJ9parr9FtvvcGI1SA': 'Chinese, Noodles',
                   'SjgeuBlgKER9yegpoxT99w': 'Lounges, Japanese, Tapas Bars', 'SW6ZQk22G1_CV81_gfnvNQ': 'Mexican',
                   'vh1tPEaPioD78QmoqnWXpw': 'Thai', 'ttuEwktrkmh3TUlSFPZqAA': 'Vietnamese, Asian Fusion',
                   'trKyIRyjKqVSZmcU0AnICQ': 'Seafood, Comfort Food, Breakfast & Brunch',
                   'B70iTJjcPkuYn8ouUewWgw': 'Italian', 'yg_A_TpYkJjr1fef0J6QkQ': 'Ramen',
                   'hdZy_F-gcW9Ntt53Sh--aQ': 'Breakfast & Brunch, Cafes',
                   'dAmVTQ6ukuLh4UxTmBoRkg': 'Egyptian, Vegan, Sandwiches', 'nBl_4gw5ecGzNkHyzfii8g': 'Italian',
                   'jc3p5SFyt9qrrMXt6E13ig': 'Tea Rooms, Ice Cream & Frozen Yogurt, Desserts',
                   'RwRNR4z3kY-4OsFqigY5sw': 'Desserts, Bakeries',
                   'htQgj-ANQpZGpIpkkrEmyQ': 'Asian Fusion, Breakfast & Brunch, French',
                   'aLcFhMe6DDJ430zelCpd2A': 'Thai', '4m_hApwQ054v3ue_OxFmGw': 'Spanish, Wine Bars, Tapas Bars',
                   'O_UC_izJXcAmkm6HlEyGSA': 'Mexican, Bars',
                   '5XVabANkehj7oH-Z7YZkwg': 'Coffee & Tea, Breakfast & Brunch',
                   'wxL9wgxLeuMMfI6SAXIzJw': 'Comfort Food, American (Traditional), Seafood',
                   'Fx5haZv9PP3E7Ljp-h7B1Q': 'Breakfast & Brunch, Juice Bars & Smoothies',
                   'wSojc-y-d7MWiGWdy8deCg': 'Barbeque, Smokehouse',
                   'i--dxuKd_6Dx7mwgQ5nl-g': 'Bakeries, Breakfast & Brunch, Canadian (New)',
                   'VRwT0pscR5vESCrAnUpNwQ': 'Mexican', 'MS-hfug4QDXqb_Mws3qlzA': 'Bars, Mexican',
                   'snw9iNNLpFYZeHotW00uVA': 'Asian Fusion',
                   'kOFDVcnj-8fd3doIpCQ06A': 'Breakfast & Brunch, Canadian (New), Vegetarian',
                   'N_2yEZ41g9zDW_gWArFiHw': 'Desserts, Ice Cream & Frozen Yogurt, Coffee & Tea',
                   'nHFJtud7jWZhM9dHQ1eIRA': 'Korean',
                   'htVvtLIFftBLqzRISjReDw': 'Desserts, American (Traditional), Salad',
                   'hDy-uY7Vy_TZdGBzw59lhA': 'Sushi Bars, Japanese', 'f5O7v_X_jCg2itqacRfxhg': 'Ramen, Noodles',
                   'MhiBpIBNTCAm1Xd3WzRzjQ': 'Greek, Sandwiches', 'fGurvC5BdOfd5MIuLUQYVA': 'Chinese',
                   'Qa4eXuZ1IFPwnVXJcpZWtw': 'Sports Bars, Canadian (New)',
                   'pSMK_FtULKiU-iuh7SMKwg': 'Pizza, Salad, Fast Food', 'RUd_M7DPJq1I3DPq0oF--w': 'Halal, Pakistani',
                   '_xAJZOKBMPOe47p1MphB2w': 'Chinese, Seafood', 'QGTqGNLZbBA1QD8L_fO9ZA': 'Japanese, Sushi Bars',
                   'Cp3YRVZojrCGeQS41Hf1pw': 'Vietnamese', 'zgQHtqX0gqMw1nlBZl2VnQ': 'Ramen',
                   'r_BrIgzYcwo1NAuG9dLbpg': 'Thai', 'SGP1jf6k7spXkgwBlhiUVw': 'Desserts, Ice Cream & Frozen Yogurt',
                   'zA6gnF5aPBGoOm6uIbKt-A': 'Sushi Bars, Japanese', '-av1lZI1JDY_RZN2eTMnWg': 'Thai',
                   'crstB-H5rOfbXhV8pX0e6g': 'Ramen', 'fzHBvd0HZm1yB3UIMqZ3bA': 'Dim Sum',
                   'CN5nuUQod0f8g3oh99qq0w': 'Japanese, Tapas/Small Plates, Pubs',
                   'h_4dPV9M9aYaBliH1Eoeeg': 'German, Bars', 'K6XIGkyk7-fuOQtA8i7p6A': 'Brewpubs',
                   'uAAWlLdsoUf872F1FKiX1A': 'Desserts, Ice Cream & Frozen Yogurt',
                   'k6zmSLmYAquCpJGKNnTgSQ': 'American (Traditional), Barbeque, Southern',
                   'HkHTdTvzbn-bmeQv_-2u0Q': 'Canadian (New), Sandwiches, Breakfast & Brunch',
                   'HUYEadSbGSQNHXFmT2Ujjw': 'Japanese, Noodles',
                   'yY3jNsrpCyKTqQuRuLV8gw': 'Cocktail Bars, Canadian (New)',
                   'piZ4JqJI5WTljJuQV7yZHQ': 'Ramen, Noodles',
                   'rxA9c0_XObabVL1WCTA4FA': 'Tex-Mex, Bars, Breakfast & Brunch'}


def read_review_dicts_with_asym(dict_subsampled, asym_reviews, asym_num=16):
    keys = list(dict_subsampled.keys())
    num_res = len(keys)
    pair_per_res = len(dict_subsampled[keys[0]])
    standard_dicts = []
    print(pair_per_res, num_res, 'asym_num', asym_num)
    asym_index = 0
    for i in range(pair_per_res):
        for j in range(num_res):
            # print(dict_subsampled[keys[j]][i])
            rev1 = dict_subsampled[keys[j]][i][0][0]
            rev2 = dict_subsampled[keys[j]][i][1][0]
            sample = {}
            sample["query"] = rev1
            sample["passages"] = [{'text': rev2, 'label': 'positive'}]
            standard_dicts.append(sample)
        for k in range(asym_num):
            sample = {}
            sample["query"] = asym_reviews[asym_index]
            asym_index += 1
            sample["passages"] = [{'text': asym_reviews[asym_index], 'label': 'positive'}]
            asym_index += 1
            standard_dicts.append(sample)

    return standard_dicts


def read_review_dicts(dict_subsampled, with_hard_negatives=False, prepend=False, prepend_both=False,
                      subsample_query=False, subsample_query_sentence=False):
    keys = list(dict_subsampled.keys())
    num_res = len(keys)
    pair_per_res = len(dict_subsampled[keys[0]])
    standard_dicts = []
    print(pair_per_res, num_res)
    for i in range(pair_per_res):
        for j in range(num_res):
            # print(dict_subsampled[keys[j]][i])
            rev1 = dict_subsampled[keys[j]][i][0][0]
            rev2 = dict_subsampled[keys[j]][i][1][0]
            sample = {}
            # TODO: filter short sentences, get a random sentence
            if subsample_query:
                l = len(rev1.split(' '))
                if l > 20:
                    a = random.randint(0, l - 20)
                    new_rev1 = ' '.join(rev1.split(' ')[a:a + 20])
                    rev1 = new_rev1
            if subsample_query_sentence:
                rev_s = sent_tokenize(rev1)
                # rev_s = [s for s in rev_s if len(s.split(' ')) > 5]
                new_rev1 = rev_s[random.randint(0, len(rev_s)-1)]
                rev1 = new_rev1
            if prepend:
                rev2 = full_categories[keys[j]] + ' ' + rev2
            if prepend_both:
                rev1 = full_categories[keys[j]] + ' ' + rev1

            sample["query"] = rev1
            # Since we are only using same restaurant for item embedding, we add business id to passage
            sample["passages"] = [{'text': rev2, 'label': 'positive'}]
            sample["business_id"] = keys[j]
            # TODO: have a map of business_id to index
            if with_hard_negatives:
                hns = dict_subsampled[keys[j]][i][2]
                for hn in hns:
                    sample["passages"].append({'text': hn, 'label': 'hard_negative'})
            standard_dicts.append(sample)
    return standard_dicts


def read_review_dicts_item(dict_subsampled):
    keys = list(dict_subsampled.keys())
    num_res = len(keys)
    pair_per_res = len(dict_subsampled[keys[0]])
    standard_dicts = []
    print(pair_per_res, num_res)
    for i in range(pair_per_res):
        for j in range(num_res):
            # print(dict_subsampled[keys[j]][i])
            rev1 = dict_subsampled[keys[j]][i]
            # rev2 = dict_subsampled[keys[j]][i]
            sample = {}
            sample["query"] = ''
            # Since we are only using same restaurant for item embedding, we add business id to passage
            sample["passages"] = [{'text': rev1, 'label': 'positive'}]
            sample["business_id"] = keys[j]
            # TODO: have a map of business_id to index
            # if with_hard_negatives:
            #     hns = dict_subsampled[keys[j]][i][2]
            #     for hn in hns:
            #         sample["passages"].append({'text': hn, 'label': 'hard_negative'})
            standard_dicts.append(sample)
    return standard_dicts


def read_review_dicts_ir_style(dict):
    keys = list(dict.keys())
    num_res = len(keys)
    # pair_per_res = len(dict[keys[0]])
    standard_dicts = []
    # print(pair_per_res, num_res)
    for j in range(num_res):
        for i in range(len(dict[keys[j]])):
            # print(dict_subsampled[keys[j]][i])
            rev1 = dict[keys[j]][i][0][0]
            rev2 = dict[keys[j]][i][1][0]
            sample = {}
            sample["query"] = rev1
            sample["passages"] = [{'text': rev2, 'label': 'positive'}]
            standard_dicts.append(sample)
    return standard_dicts
