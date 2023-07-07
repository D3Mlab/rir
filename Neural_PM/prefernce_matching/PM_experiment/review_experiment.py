# Mahdi Abdollahpour, 21/03/2022, 09:27 PM, PyCharm, lgeconvrec

from Neural_PM.utils.exp import *
from Neural_PM.utils.eval import *
from Neural_PM.prefernce_matching.matrix import *
from Neural_PM.utils.load_data import *
from Neural_PM.prefernce_matching.PM.item import *
from Neural_PM.prefernce_matching.PM.review import *
from Neural_PM.prefernce_matching.PM.hybrid import *
import os
import pandas as pd

from Neural_PM.prefernce_matching.PM_experiment.experiment_utils import *
from Neural_PM.clustering.vectorization import TFIDFVectorizerModel

def run_experiments(create_matrices=True, BERT_name='MSMARCO',
                    save_path='matrices/',
                    agg_method='topk',
                    k=10,
                    split_sentence=True,
                    sentence_agg_method='topk',
                    sentence_k=5,
                    save_output=True,
                    save_metrics=True, rate_size=1, cosine=False, seed=None, subsample=False,
                    results_save_path='./results/generated_result/',
                    true_labels_path='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv',
                    filtered_review_data='./data/400_review_3_star_above - 400_review_3_star_above.csv', strategy=None,
                    from_pt=False, settings_to_include=None, prepend_categories=False, tf_idf=False,add_details_to_path=True):
    '''
    Get the setting and run once
    :param create_matrices:If True starts making them if False reads from saved
    :param BERT_name:
    :param save_path:
    :param agg_method:Top k or average for review
    :param k: k for top k in review
    :param split_sentence: True or False: take all sentences
    :param sentence_agg_method:
    :param sentence_k:
    :param save_output:For false positive detection from create_output_text()
    :param save_metrics:To save or not the metrics
    :param rate_size:For seeing the amount of reviews per restuarant
    :param cosine:True: Cosine, False: Dot product
    :param seed:For randomness in subsampling
    :param subsample:if True:subsampled, If False: not sub sampled and whole data
    :param results_save_path:
    :return:
    eval_result: A dictionary with metric name as keys and the values and values -> like process_results.process()
    output_message: Example:
                    #1 --- Query:Can I have a cheat meal?
                    Best Matching Resturant:Byblos
                    Best Matching Reviews of the Resturant:
                    #1--- score: 166.00059428740994 - It's really up to personal preference. I am not a big fun. Portion is too small. Fig salad is highly recommended. A little bit disappointed.
                    #2--- score: 165.7939427449958 - Ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh­hhhhhhhhhhhhhhhhhhhhhhSo incredibly. Best meal in years. nough said
                    First relevant at rank of 1 (starting from 1), name: Byblos --> This case it is True but it can be False positive and another restaurant
                    a review from that:
                    score: 166.00059428740994 - It's really up to personal preference. I am not a big fun. Portion is too small. Fig salad is highly recommended. A little bit disappointed.
    '''
    sim = 'dot'
    if cosine:
        sim = 'cosine'
    if tf_idf:
        save_path = './tf_idf'
        BERT_name = 'TF-IDF'
    stripped_BN = BERT_name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    if add_details_to_path:
        if split_sentence:
            EXP_NAME = "_".join([stripped_BN, agg_method, str(k), sentence_agg_method, str(sentence_k),
                                 'size', str(rate_size), sim, str(seed), str(subsample)])
        else:
            EXP_NAME = "_".join([stripped_BN, agg_method, str(k), 'NA', 'NA', 'size', str(rate_size), sim,
                                 str(seed), str(subsample)])
    else:
        EXP_NAME = stripped_BN

    df = pd.read_csv(filtered_review_data)
    if prepend_categories:
        df['review_text'] = df['categories'] + ' ' + df['review_text']
    df['index'] = df.index
    restaurants = df.groupby(['name'])
    print(df.shape)

    true_labels = build_true_labels(true_labels_path)
    queries, restaurants = get_queries_and_resturants(true_labels_path)
    restaurants = restaurants.tolist()

    reviews, id_name_map = sort_reviews_by_business(df, restaurants)
    indexes, id_name_map = sort_reviews_by_business(df, restaurants, 'index')
    print(len(reviews), len(indexes))

    if create_matrices or not os.path.exists(save_path):
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        if tf_idf:
            embedder = TFIDFVectorizerModel()

        else:
            if strategy is not None:
                with strategy.scope():
                    embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)
            else:
                embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)
        if not tf_idf:
            save_matrices(queries, reviews, save_path, embedder, split_sentence=split_sentence, strategy=strategy)
        else:
            save_matrices_tfidf(queries, reviews, save_path, embedder, split_sentence=split_sentence)

    model = ReviewPreferenceMatching(restaurants=restaurants, id_name_map=id_name_map, save_directory=save_path,
                                     split_sentence=split_sentence, cosine=cosine, n=rate_size, seed=seed,
                                     subsample=subsample)

    outputs, best_matching_reviews = model.predict(sentence_agg_k=sentence_k, agg_method=agg_method, k=k,
                                                   sentence_agg_method=sentence_agg_method)

    query_score = outputs

    query_score_ranked = ranking(query_score)

    name_to_id = {v: k for k, v in id_name_map.items()}

    output_message = create_output_text(queries, best_matching_reviews, name_to_id, restaurants, query_score_ranked,
                                        true_labels)

    ev_res = {}
    ev_res['similarity'] = sim
    ev_res['BERT prefernce_matching'] = BERT_name
    ev_res['Review aggregation'] = agg_method
    if agg_method != 'avg':
        ev_res['k_R'] = k
    else:
        ev_res['k_R'] = 'NA'
    if split_sentence:
        ev_res['Sentence aggregation'] = sentence_agg_method
        ev_res['k_S'] = sentence_k
    else:
        ev_res['Sentence aggregation'] = 'NA'
        ev_res['k_S'] = 'NA'

    eval_result = evaluation_prediction(true_labels, query_score_ranked)
    eval_result_pq = per_query_result(true_labels, query_score_ranked)
    by_q_type = evaluate_by_query_type(true_labels, query_score_ranked)
    eval_result['by_query_type'] = by_q_type
    for key in eval_result:
        ev_res[key] = eval_result[key]

    ev_res['size'] = rate_size
    ev_res['subsample'] = str(subsample)
    ev_res['seed'] = str(seed)

    if settings_to_include is not None:
        for key in settings_to_include:
            ev_res[key] = str(settings_to_include[key])

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    if save_output:
        with open(os.path.join(results_save_path, EXP_NAME + '_output.txt'), mode='w', encoding='utf8') as f:
            f.write(output_message)
    if save_metrics:
        with open(os.path.join(results_save_path, EXP_NAME + '_metrics.txt'), mode='w', encoding='utf8') as f:
            f.write(str(ev_res))
    if save_metrics:
        with open(os.path.join(results_save_path, EXP_NAME + '_perquery.txt'), mode='w', encoding='utf8') as f:
            f.write(str(eval_result_pq))
    return eval_result, output_message
