# Mahdi Abdollahpour, 16/03/2022, 10:53 AM, PyCharm, Neural_PM

from Neural_PM.prefernce_matching.PM import *
from Neural_PM.prefernce_matching.statics import TOEKNIZER_MODELS, BERT_MODELS

def add_cosine(settings):
    new_settings = []
    for s in settings:
        copy = s.copy()
        copy['cosine'] = True
        new_settings.append(s)
        new_settings.append(copy)
    return new_settings


def get_settings(cosine=False, bert_names=['MSMARCO'], split_sentence=[True, False], k_ranges_agg=[1, 5, 10, 20],
                 agg_methods=['topk', 'avg'],
                 filtered_review_data='./data/400_review_3_star_above - 400_review_3_star_above.csv',
                 true_labels_path='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv',
                 results_save_path='./generated_results'):
    '''
    Makes all the different settings for experiments.
    :param cosine: False -> no cosine, True -> Cosine
    :param bert_names: A key of the bert prefernce_matching you want from the dictionary in PM.py, local or from hugging face
    :param split_sentence: If we split or not
    :param k_ranges_agg: top k aggregation method
    :param agg_methods: top k or average of all
    :return: List of experiments, each experiment is a dictionary like this:
        EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 4,  # for review aggregation
        'split_sentence': True,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 5,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'cosine': cosine,
    }
    '''
    EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 4,  # for review aggregation
        'split_sentence': True,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 5,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'results_save_path': results_save_path,
        'cosine': cosine,
        'true_labels_path': true_labels_path,
        'filtered_review_data': filtered_review_data,
        'strategy': None,
    }

    # split_sentence = [False]
    '''
    BERT

    '''
    setup_list = []

    for value in split_sentence:

        if not value:  # No sentence aggregation

            EXP_SETTING['split_sentence'] = False
            for i, BERT in enumerate(bert_names):
                EXP_SETTING['BERT_name'] = BERT
                EXP_SETTING['save_path'] = './uot_review_based_recommendation/' + BERT + '_split_' + str(value) + '/'

                for agg_method in agg_methods:
                    if agg_method == 'topk':
                        EXP_SETTING['agg_method'] = 'topk'
                        for k_rev in k_ranges_agg:
                            EXP_SETTING['k'] = k_rev
                            setup_list.append(EXP_SETTING.copy())
                    elif agg_method == 'avg':
                        EXP_SETTING['agg_method'] = 'avg'
                        setup_list.append(EXP_SETTING.copy())

        else:  # Sentece aggregation

            EXP_SETTING['split_sentence'] = True
            for i, BERT in enumerate(bert_names):

                EXP_SETTING['BERT_name'] = BERT
                EXP_SETTING['BERT_name'] = BERT
                EXP_SETTING['save_path'] = './uot_review_based_recommendation/' + BERT + '_split_' + str(value) + '/'

                for agg_method in agg_methods:
                    if agg_method == 'topk':
                        EXP_SETTING['agg_method'] = 'topk'
                        for k_rev in k_ranges_agg:
                            EXP_SETTING['k'] = k_rev
                            for sen_agg_method in agg_methods:
                                if sen_agg_method == 'topk':
                                    EXP_SETTING['sentence_agg_method'] = 'topk'
                                    for k_sen in k_ranges_agg:
                                        EXP_SETTING['sentence_k'] = k_sen
                                        setup_list.append(EXP_SETTING.copy())
                                elif sen_agg_method == 'avg':
                                    EXP_SETTING['sentence_agg_method'] = 'avg'
                                    setup_list.append(EXP_SETTING.copy())

                    elif agg_method == 'avg':
                        EXP_SETTING['agg_method'] = 'avg'
                        for sen_agg_method in agg_methods:
                            if sen_agg_method == 'topk':
                                EXP_SETTING['sentence_agg_method'] = 'topk'
                                for k_sen in k_ranges_agg:
                                    EXP_SETTING['sentence_k'] = k_sen
                                    setup_list.append(EXP_SETTING.copy())
                            elif sen_agg_method == 'avg':
                                EXP_SETTING['sentence_agg_method'] = 'avg'
                                setup_list.append(EXP_SETTING.copy())

    return setup_list




def get_short_paper_settings(bert_names=['MSMARCO'], filtered_review_data='./data/400_review_3_star_above - 400_review_3_star_above.csv',
                 true_labels_path='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv',
                 results_save_path='./generated_results'):
    '''
    Makes all the different settings for experiments.
    :param cosine: False -> no cosine, True -> Cosine
    :param bert_names: A key of the bert prefernce_matching you want from the dictionary in PM.py, local or from hugging face
    :param split_sentence: If we split or not
    :param k_ranges_agg: top k aggregation method
    :param agg_methods: top k or average of all
    :return: List of experiments, each experiment is a dictionary like this:
        EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 4,  # for review aggregation
        'split_sentence': True,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 5,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'cosine': cosine,
    }
    '''
    EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 1,  # for review aggregation
        'split_sentence': False,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 1,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'results_save_path': results_save_path,
        'cosine': False,
        'true_labels_path': true_labels_path,
        'filtered_review_data': filtered_review_data,
        'strategy': None,
    }

    # split_sentence = [False]
    '''
    BERT

    '''
    setup_list = []


    EXP_SETTING['split_sentence'] = False
    for i, BERT in enumerate(bert_names):
        EXP_SETTING['BERT_name'] =BERT
        EXP_SETTING['save_path'] = './short_paper/' + BERT +'/'


        EXP_SETTING['agg_method'] = 'topk'

        EXP_SETTING['k'] = 1
        setup_list.append(EXP_SETTING.copy())



    return setup_list




def get_debug_settings(k_R=20, k_S=[3, 10]):
    setup_list = []
    for k in k_S:
        EXP_SETTING = {
            'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

            'create_matrices': False,  # if true always creats, if false checks if not exits creates
            'save_path': './uot_review_based_recommendation/VANILLA_split_True/',
            # choose a uniqe name between diffrent bert,split settings
            'agg_method': 'topk',  # one of 'topk','avg','max'
            'k': k_R,  # for review aggregation
            'split_sentence': True,
            'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
            'sentence_k': k,  # only used when split_sentence=True
            'save_output': True,  # save to file
            'save_metrics': True,  # save to file
            'rate_size': 1,
            'cosine': True,
        }
        setup_list.append(EXP_SETTING.copy())

    return setup_list


def get_ratio_settings(cosine=False):
    '''
    Instead of TOP-k method aggregating the top k% of the best match review(not sentence)
    :param cosine:
    :return: List of settings with a dictionay per setting like:
            EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 4,  # for review aggregation
        'split_sentence': True,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 5,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'cosine': cosine,
    }
    '''
    EXP_SETTING = {
        'BERT_name': 'VANILLA',  # use keys of BERT_MODELS dictionary

        'create_matrices': False,  # if true always creats, if false checks if not exits creates
        'save_path': 'msmarco_matrices_non_split/',  # choose a uniqe name between diffrent bert,split settings
        'agg_method': 'topk',  # one of 'topk','avg','max'
        'k': 4,  # for review aggregation
        'split_sentence': True,
        'sentence_agg_method': 'topk',  # only used when split_sentence=True, one of 'topk','avg','max'
        'sentence_k': 5,  # only used when split_sentence=True
        'save_output': True,  # save to file
        'save_metrics': True,  # save to file
        'rate_size': 1,
        'cosine': cosine,
        'results_save_path': './ratio_results/',
    }

    k_ranges_agg = [0.90, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    agg_methods = ['ratio']
    split_sentence = [False]
    # split_sentence = [False]
    '''
    BERT

    '''
    setup_list = []
    for value in split_sentence:

        if not value:  # No sentence aggregation

            EXP_SETTING['split_sentence'] = False
            for i, BERT in enumerate(list(BERT_MODELS.keys())[:1]):
                EXP_SETTING['BERT_name'] = BERT
                EXP_SETTING['save_path'] = './uot_review_based_recommendation/' + BERT + '_split_' + str(value) + '/'

                for agg_method in agg_methods:
                    if agg_method == 'ratio':
                        EXP_SETTING['agg_method'] = 'ratio'
                        for k_rev in k_ranges_agg:
                            EXP_SETTING['k'] = k_rev
                            setup_list.append(EXP_SETTING.copy())

    return setup_list
