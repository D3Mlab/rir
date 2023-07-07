# Mahdi Abdollahpour, 16/03/2022, 10:48 AM, PyCharm, Neural_PM

import pandas as pd
import numpy as np
import scipy


def build_true_labels(path_of_labels):
    df = pd.read_csv(path_of_labels)
    queries = df["query"].unique()
    restaurants = df["Restaurant name"].unique()
    final_df = pd.DataFrame(index=queries, columns=restaurants)
    for index, row in df.iterrows():
        final_df.at[row["query"], row["Restaurant name"]] = row["If only Low or  High"]
    # return final_df.to_numpy(),final_df.columns,final_df.index
    return final_df.values


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def ap_atk(final_relevance_ranked, k):
    prec = []
    for i in range(final_relevance_ranked.shape[0]):
        all_relevant_till_k = np.sum(final_relevance_ranked[i, :k])
        prec.append(all_relevant_till_k / k)
    return np.average(prec), mean_confidence_interval(prec)


def recall_atk(final_relevance_ranked, k):
    recall = []
    for i in range(final_relevance_ranked.shape[0]):
        all_relevant = np.sum(final_relevance_ranked[i,])
        if all_relevant == 0:
            print('no relevent Item for query number ', i)
            continue
        all_relevant_till_k = np.sum(final_relevance_ranked[i, :k])
        recall.append(all_relevant_till_k / all_relevant)
    return np.average(recall), mean_confidence_interval(recall)


def r_prec(final_relevance_ranked):
    r_precs = []
    for i in range(final_relevance_ranked.shape[0]):
        r = np.sum(final_relevance_ranked[i,])
        r_precs.append(np.sum(final_relevance_ranked[i, :int(r)] / r))
    a_list = r_precs
    converted_list = [float(element) for element in a_list]
    # txt = ",".join(converted_list)
    return np.average(r_precs), mean_confidence_interval(r_precs), converted_list


def map(final_relevance_ranked):
    prec = []
    for i in range(final_relevance_ranked.shape[0]):
        prec_per_query = []
        for j in range(1, final_relevance_ranked.shape[1] + 1):
            if final_relevance_ranked[i, j - 1] == 1:
                prec_per_query.append(ap_atk(final_relevance_ranked[i, :].reshape(1, -1), j)[0])

        prec.append(np.sum(prec_per_query) / np.sum(final_relevance_ranked[i, :]))
    a_list = prec
    converted_list = [float(element) for element in a_list]
    # txt = ",".join(converted_list)
    return np.average(prec), mean_confidence_interval(prec), converted_list


def evaluate_by_query_type(true_labels, prediction_ranking):
    results = {}
    results['Indirect queries'] = evaluation_prediction(true_labels[:20, :], prediction_ranking[:20, :], False, False)
    results['queries with negation'] = evaluation_prediction(true_labels[20:40, :], prediction_ranking[20:40, :], False,
                                                             False)
    results['genral queries'] = evaluation_prediction(true_labels[40:60, :], prediction_ranking[40:60, :], False, False)
    results['detailed queries'] = evaluation_prediction(true_labels[60:80, :], prediction_ranking[60:80, :], False,
                                                        False)
    results['contradictory queries'] = evaluation_prediction(true_labels[80:, :], prediction_ranking[80:, :], False,
                                                             False)
    return results


def per_query_result(true_labels, prediction_ranking):
    '''
    Inputs:
      true_labels: the real labels for each query in a numpy matrix format of Queries x Restaurant: which 1s means relevant
      prediction_ranking: the descending sorted index of restaurant for each query
      atk: The number of K for metrics
    Output a dictionary with evaluation metrics and corresponding values
    '''
    final_relevance_ranked = np.zeros((true_labels.shape[0], true_labels.shape[1]))
    for i in range(true_labels.shape[0]):
        for j in range(true_labels.shape[1]):
            final_relevance_ranked[i, j] = true_labels[i, prediction_ranking[i, j]]

    results = {}
    results['R_Prec'] = r_prec(final_relevance_ranked)[2]
    results['MAP'] = map(final_relevance_ranked)[2]
    return results


def evaluation_prediction(true_labels, prediction_ranking, do_for_ks=True, add_all_R_Prec=True):
    '''
    Inputs:
      true_labels: the real labels for each query in a numpy matrix format of Queries x Restaurant: which 1s means relevant
      prediction_ranking: the descending sorted index of restaurant for each query
      atk: The number of K for metrics
    Output a dictionary with evaluation metrics and corresponding values
    '''
    final_relevance_ranked = np.zeros((true_labels.shape[0], true_labels.shape[1]))
    for i in range(true_labels.shape[0]):
        for j in range(true_labels.shape[1]):
            final_relevance_ranked[i, j] = true_labels[i, prediction_ranking[i, j]]

    eval_ks = [1, 3, 4, 10]
    results = {'R-Prec': r_prec(final_relevance_ranked)[0],
               'R-Prec_CI': r_prec(final_relevance_ranked)[1],
               'MAP': map(final_relevance_ranked)[0],
               'MAP_CI': map(final_relevance_ranked)[1],
               }
    if add_all_R_Prec:
        results['all_R_Prec'] = r_prec(final_relevance_ranked)[2]
    if do_for_ks:
        for i in eval_ks:
            r = recall_atk(final_relevance_ranked, i)
            results['Recall@' + str(i)] = r[0]
            results['Recall@' + str(i) + '_CI'] = r[1]
            ap = ap_atk(final_relevance_ranked, i)
            results['AP@' + str(i)] = ap[0]
            results['AP@' + str(i) + '_CI'] = ap[1]

    # results = {
    #     '1': {'Recall @1': recall_atk(final_relevance_ranked, 1)[0],
    #           'Average Precision @1': ap_atk(final_relevance_ranked, 1)[0],
    #           'R-Prec': r_prec(final_relevance_ranked)[0],
    #           'Recall @1_CI': recall_atk(final_relevance_ranked, 1)[1],
    #           'Average Precision @1_CI': ap_atk(final_relevance_ranked, 1)[1],
    #           'R-Prec_CI': r_prec(final_relevance_ranked)[1],
    #           'MAP': map(final_relevance_ranked)[0],
    #           'MAP CI': map(final_relevance_ranked)[1],
    #           'all_R_Prec': r_prec(final_relevance_ranked)[2]
    #
    #           },
    #
    #     '3': {'Recall @3': recall_atk(final_relevance_ranked, 3)[0],
    #           'Average Precision @3': ap_atk(final_relevance_ranked, 3)[0],
    #           'R-Prec': r_prec(final_relevance_ranked)[0],
    #           'Recall @3_CI': recall_atk(final_relevance_ranked, 3)[1],
    #           'Average Precision @3_CI': ap_atk(final_relevance_ranked, 3)[1],
    #           'R-Prec_CI': r_prec(final_relevance_ranked)[1], },
    #
    #     '5': {'Recall @5': recall_atk(final_relevance_ranked, 5)[0],
    #           'Average Precision @5': ap_atk(final_relevance_ranked, 5)[0],
    #           'R-Prec': r_prec(final_relevance_ranked)[0],
    #           'Recall @5_CI': recall_atk(final_relevance_ranked, 5)[1],
    #           'Average Precision @5_CI': ap_atk(final_relevance_ranked, 51)[1],
    #           'R-Prec_CI': r_prec(final_relevance_ranked)[1]},
    #
    #     '10': {'Recall @10': recall_atk(final_relevance_ranked, 10)[0],
    #            'Average Precision @10': ap_atk(final_relevance_ranked, 10)[0],
    #            'R-Prec': r_prec(final_relevance_ranked)[0],
    #            'Recall @10_CI': recall_atk(final_relevance_ranked, 10)[1],
    #            'Average Precision @10_CI': ap_atk(final_relevance_ranked, 10)[1],
    #            'R-Prec_CI': r_prec(final_relevance_ranked)[1]}
    #
    # }
    return results


def ranking(scores_matrix):
    sorted = np.argsort(scores_matrix, axis=1)
    return np.flip(sorted, 1)
