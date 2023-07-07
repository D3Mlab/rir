# Mahdi Abdollahpour, 08/04/2022, 09:42 AM, PyCharm, lgeconvrec
from sklearn.utils import shuffle

from Neural_PM.finetune.train import *
from Neural_PM.prefernce_matching.PM_experiment import item_experiment, review_experiment, hybrid_experiment
from Neural_PM.utils.process_results import process
from Neural_PM.finetune.sampling.sampling import *
from Neural_PM.prefernce_matching.statics import BERT_MODELS, TOEKNIZER_MODELS
from Neural_PM.clustering.clustering import *
from Neural_PM.clustering.clustering_experiment import run_clustering
from Neural_PM.utils.eval import mean_confidence_interval
import numpy as np
import time
from datetime import datetime
import time
from Neural_PM.finetune.train_utils import setup_tpu

from Neural_PM.finetune.data_loading import *


def run(finetune_setting, val_ratio=0.2):
    '''

    :param finetune_setting:
    :param val_ratio:
    :param finetune_function:
    -Declare experiment paths with the setting name to save LM and results
    -Getting the positive sample for val and train
    -fine_tune function : review_finetune/ item_embedding --> save LM in path
    - Run_PM: to test LM, read LM from path where it was saved from fine-tune function
    :return:
    '''
    stripped_BN = finetune_setting['model_name'].replace('./', '#').replace('/', '#').replace('-', '#').replace('_',
                                                                                                                '$')
    EXP_name = str(finetune_setting['number']) + '_finetune_' + 'IR(' + str(finetune_setting['ir_style'])[0] + ')_' + \
               str(stripped_BN)[:5] + \
               '_SR(' + str(finetune_setting['same_rating'])[0] + ')' + \
               '_LS(' + str(finetune_setting['least_similar'])[0] + ')' + \
               '_HN(' + str(finetune_setting['hard_negative'])[0] + ')' + \
               '_SQ(' + str(finetune_setting['subsample_query'])[0] + ')' + \
               '_SQS(' + str(finetune_setting['subsample_query_sentence'])[0] + ')' + \
               '_PC(' + str(finetune_setting['prepend_neuralpm'])[0] + ')' + \
               '_' + (str(time.asctime(time.localtime(time.time()))).replace(' ', '-'))
    # +'_' + str(time.time() * 10000000)
    # TODO: fix time stamp
    finetune_setting['save_path'] = os.path.join('finetune', EXP_name)
    if finetune_setting['item_embedding']:
        finetune_function = item_embedding_finetune
    else:
        finetune_function = review_finetune
    if not os.path.exists(finetune_setting['save_path']):
        os.makedirs(finetune_setting['save_path'])

    with open(os.path.join(finetune_setting['save_path'], 'finetune_setting.json'), 'wb') as f:
        f.write(str.encode(str(finetune_setting)))

    if finetune_setting['tpu']:
        strategy = setup_tpu()
        BS = finetune_setting['batch_size_per_replica'] * strategy.num_replicas_in_sync
    else:
        strategy = None
        BS = finetune_setting['batch_size_per_replica']

    if finetune_setting["asym_negative"]:
        assert BS > 50, "no need for Asym when batch size < 50"
        finetune_setting["asym_num"] = BS - 50
    if finetune_setting["sym_negative"]:
        assert BS > 50, "no need for Sym when batch size < 50"
        # finetune_setting["sym_num"] = BS - 50

    if finetune_setting['epochs'] > 0:
        dicts_standard_train, dicts_standard_val, keys = get_positive_samples(finetune_setting, val_ratio=val_ratio)

        # data_len = len(dicts_standard)
        # train_len = int(data_len * (1 - val_ratio))
        # dicts_standard_train = dicts_standard[:train_len]
        # dicts_standard_val = dicts_standard[train_len:]
        finetune_setting['keys'] = keys
        finetune_function(dicts_standard_train, dicts_standard_val, finetune_setting, strategy)
        # Saves path of the LM and run_pm() reads that file for testing
    if finetune_setting['run_neuralpm']:
        # TODO: pass the language model object
        # reads LM from path
        if finetune_setting['item_embedding']:
            results_pd = run_pm_with_item(finetune_setting, strategy)
        else:
            results_pd = run_pm(finetune_setting, strategy)
        return results_pd
    return None


def run_pm(finetune_setting, strategy=None):
    save_path = finetune_setting['save_path']
    results_path = os.path.join(save_path, 'generated_result')
    bn = os.path.join(save_path, 'LM')
    BERT_MODELS[bn] = bn
    TOEKNIZER_MODELS[bn] = finetune_setting['model_name']
    settings_to_include = {}
    keys = ['same_rating', 'learning_rate', 'batch_size_per_replica', 'temperature', 'patience', 'ir_style',
            'least_similar', 'most_similar', 'hard_negative',
            'hard_negative_num', 'number', 'embedded_reviews_path', 'positive_pair_per_restaurant']
    for key in keys:
        settings_to_include[key] = finetune_setting[key]
    if not finetune_setting['hard_negative']:
        settings_to_include['hard_negative_num'] = 'NA'
    if not (finetune_setting['least_similar'] or finetune_setting['most_similar']):
        settings_to_include['embedded_reviews_path'] = 'NA'
    pm_setting = {
        'create_matrices': not finetune_setting['reuse_matrices'],
        'BERT_name': bn, 'save_path': os.path.join(save_path, 'matrices'), 'agg_method': 'topk', "k": 1,
        'split_sentence': False, 'sentence_agg_method': 'topk', 'sentence_k': 5, 'save_output': True,
        'save_metrics': True, 'rate_size': 1, 'cosine': False, 'seed': None, 'subsample': False,
        'results_save_path': results_path, 'true_labels_path': finetune_setting['true_labels_path'],
        'filtered_review_data': finetune_setting['filtered_review_data'], 'strategy': strategy,
        'settings_to_include': settings_to_include, 'prepend_categories': finetune_setting['prepend_neuralpm'],
    }
    k_set = [1, 5, 10, 20]
    for i, k in enumerate(k_set):
        setting = pm_setting.copy()
        setting['k'] = k
        if i > 0:
            setting['create_matrices'] = False
        o, _ = run_experiments(**setting)
    setting = pm_setting.copy()
    setting['agg_method'] = 'avg'
    setting['create_matrices'] = False
    o, _ = run_experiments(**setting)

    directory = results_path
    results_path = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith("metrics.txt"):
            results_path.append(f)

    #
    results_pd = process(results_path)
    # print(results_pd)
    # print(results_pd['R-Prec'])
    # print(results_pd['MAP'])
    results_pd.to_csv(os.path.join(save_path, 'results.csv'), index=False)
    print(results_pd[['Review aggregation', 'k_R', 'R-Prec', 'MAP']].to_markdown())
    return results_pd


def run_pm_with_item(finetune_setting, strategy=None):
    save_path = finetune_setting['save_path']
    results_path = os.path.join(save_path, 'generated_result')
    item_embeddings_path = os.path.join(save_path, 'item_embeddings.npy')
    id_to_index_path = os.path.join(save_path, 'id_to_index.json')
    bn = os.path.join(save_path, 'LM')
    BERT_MODELS[bn] = bn
    TOEKNIZER_MODELS[bn] = finetune_setting['model_name']

    settings_to_include = {}
    keys = ['item_embedding', 'freeze_lm', 'same_rating', 'learning_rate', 'batch_size_per_replica', 'temperature',
            'patience', 'ir_style',
            'least_similar', 'most_similar', 'hard_negative',
            'hard_negative_num', 'number', 'embedded_reviews_path', 'positive_pair_per_restaurant']
    for key in keys:
        settings_to_include[key] = finetune_setting[key]

    pm_setting = {

        'item_embeddings_path': item_embeddings_path,
        'id_to_index_path': id_to_index_path,
        'create_matrices': not finetune_setting['reuse_matrices'],
        'BERT_name': bn, 'save_path': os.path.join(save_path, 'matrices'),
        'save_metrics': True, 'cosine': False, 'seed': None,
        'results_save_path': results_path, 'true_labels_path': finetune_setting['true_labels_path'],
        'filtered_review_data': finetune_setting['filtered_review_data'], 'strategy': strategy,
        'settings_to_include': settings_to_include, 'prepend_categories': finetune_setting['prepend_neuralpm'],
    }
    o, _ = run_experiments_with_item_embedding(**pm_setting)

    directory = results_path
    results_path = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith("metrics.txt"):
            results_path.append(f)

    #
    results_pd = process(results_path)
    # print(results_pd)
    # TODO: print result
    # print(results_pd['R-prec'])
    # print(results_pd['MAP'])
    results_pd.to_csv(os.path.join(save_path, 'results.csv'), index=False)
    print(results_pd[['R-Prec', 'MAP']].to_markdown())

    return results_pd


def run_upperbound(finetune_setting, val_ratio=0.2, finetune_function=upperbound_finetune):
    stripped_BN = finetune_setting['model_name'].replace('./', '#').replace('/', '#').replace('-', '#').replace('_',
                                                                                                                '$')
    EXP_name = str(finetune_setting['number']) + '_upperbound_' + str(
        finetune_setting['positive_pair_per_restaurant']) + '_' + \
               str(finetune_setting['batch_size_per_replica']) + '_' + str(
        finetune_setting['learning_rate']) + '_' + stripped_BN
    #        + \
    #        + '_asym_' + str(
    # finetune_setting['asym_negative']) + '_samerating_' + str(finetune_setting['same_rating']) + '_sym_' + str(
    # finetune_setting['sym_negative']) + str(
    # finetune_setting['sym_negative_num']) + '_hightfidf_' + str(finetune_setting['high_tfidf'])

    finetune_setting['save_path'] = './' + EXP_name

    if not os.path.exists(finetune_setting['save_path']):
        os.mkdir(finetune_setting['save_path'])

    with open(os.path.join(finetune_setting['save_path'], 'finetune_setting.json'), 'wb') as f:
        f.write(str.encode(str(finetune_setting)))

    if finetune_setting['tpu']:
        strategy = setup_tpu()
        BS = finetune_setting['batch_size_per_replica'] * strategy.num_replicas_in_sync
    else:
        strategy = None
        BS = finetune_setting['batch_size_per_replica']

    if finetune_setting["asym_negative"]:
        assert BS > 50, "no need for Asym when batch size < 50"
        finetune_setting["asym_num"] = BS - 50
    if finetune_setting["sym_negative"]:
        assert BS > 50, "no need for Sym when batch size < 50"
        # finetune_setting["sym_num"] = BS - 50

    if finetune_setting['epochs'] > 0:
        train_reviews, val_reviews, all_reviews = get_train_val_dfs(finetune_setting["train_data_path"],
                                                                    finetune_setting)
        restaurants = all_reviews.business_id.unique()
        dicts_standard_train, dicts_standard_val = get_positive_samples_using_upper_bound(all_reviews, val_reviews,
                                                                                          restaurants, finetune_setting[
                                                                                              "true_labels_path"],
                                                                                          finetune_setting[
                                                                                              "hard_negative_num"],
                                                                                          seed=finetune_setting['seed'],
                                                                                          rev_from_res=finetune_setting[
                                                                                              'rev_from_res'])
        dicts_standard_train = read_review_dicts_upper_bound(dicts_standard_train)
        dicts_standard_val = read_review_dicts_upper_bound(dicts_standard_val)
        print('Train data num', len(dicts_standard_train))
        # print('Val data num', len(dicts_standard_val))
        finetune_function(dicts_standard_train, dicts_standard_val, finetune_setting, strategy)
    if finetune_setting['run_neuralpm']:
        # TODO: pass the language model object
        results_pd = run_pm(finetune_setting, strategy)
        return results_pd
    return None
