from Neural_PM.utils.exp import *
from Neural_PM.utils.eval import *
from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.utils.load_data import *
from Neural_PM.prefernce_matching.PM.item import *
from Neural_PM.prefernce_matching.PM.review import *
from Neural_PM.prefernce_matching.PM.hybrid import *
import os
import pandas as pd


def create_output_text(queries, best_matching_reviews, name_to_id, restaurants, query_score_ranked, true_labels):
    '''
    For failure analysis to get the top false positivly matched reviews to query
    :param queries:
    :param best_matching_reviews:
    :param name_to_id:
    :param restaurants:
    :param query_score_ranked:
    :param true_labels:
    :return:
    Example:
                    #1 --- Query:Can I have a cheat meal?
                    Best Matching Resturant:Byblos
                    Best Matching Reviews of the Resturant:
                    #1--- score: 166.00059428740994 - It's really up to personal preference. I am not a big fun. Portion is too small. Fig salad is highly recommended. A little bit disappointed.
                    #2--- score: 165.7939427449958 - Ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhSo incredibly. Best meal in years. nough said
                    First relevant at rank of 1 (starting from 1), name: Byblos --> This case it is True but it can be False positive and another restaurant
                    a review from that:
                    score: 166.00059428740994 - It's really up to personal preference. I am not a big fun. Portion is too small. Fig salad is highly recommended. A little bit disappointed.
    '''
    output_message = ""
    for i in range(len(queries)):
        output_message += '#' + str(i + 1) + ' --- Query:' + queries[i] + '\n'
        output_message += 'Best Matching Resturant:' + restaurants[query_score_ranked[i, 0]] + '\n'
        output_message += 'Best Matching Reviews of the Resturant:' + '\n'
        rev_list = best_matching_reviews[i][name_to_id[restaurants[query_score_ranked[i, 0]]]]
        for idx, r in enumerate(rev_list[:2]):
            output_message += '#' + str(idx + 1) + '--- score: ' + str(r[1]) + ' - ' + str(r[0]) + '\n'

        for j in range(len(restaurants)):
            if true_labels[i, query_score_ranked[i, j]] == 1:
                output_message += 'First relevant at rank of ' + str(j + 1) + ' (starting from 1), name: ' + \
                                  restaurants[query_score_ranked[i, j]] + '\n'
                rev_list = best_matching_reviews[i][name_to_id[restaurants[query_score_ranked[i, j]]]]
                output_message += 'a review from that: \n' + 'score: ' + str(rev_list[0][1]) + ' - ' + str(
                    rev_list[0][0]) + '\n'
                break
        output_message += '-' * 100 + '\n'

    return output_message

