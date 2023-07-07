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


def run_experiments_hybrid(item_embeddings_path, id_to_index_path, create_matrices=True, BERT_name='MSMARCO',
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
                           filtered_review_data='./data/400_review_3_star_above - 400_review_3_star_above.csv',
                           strategy=None,
                           from_pt=False, settings_to_include=None, prepend_categories=False):

    sim = 'dot'
    if cosine:
        sim = 'cosine'
    stripped_BN = BERT_name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    if split_sentence:
        EXP_NAME = "_".join([stripped_BN, agg_method, str(k), sentence_agg_method, str(sentence_k),
                             'size', str(rate_size), sim, str(seed), str(subsample)])
    else:
        EXP_NAME = "_".join([stripped_BN, agg_method, str(k), 'NA', 'NA', 'size', str(rate_size), sim,
                             str(seed), str(subsample)])

    df = pd.read_csv(filtered_review_data)
    if prepend_categories:
        df['review_text'] = df['categories'] + ' ' + df['review_text']
    restaurants = df.groupby(['name'])
    print(df.shape)

    true_labels = build_true_labels(true_labels_path)
    queries, restaurants = get_queries_and_resturants(true_labels_path)
    restaurants = restaurants.tolist()

    reviews, id_name_map = sort_reviews_by_business(df, restaurants)
    print(len(reviews))

    if create_matrices or not os.path.exists(save_path):
        BERT_model_name = BERT_MODELS[BERT_name]
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        if strategy is not None:
            with strategy.scope():
                embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)
        else:
            embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)

        save_matrices(queries, reviews, save_path, embedder, split_sentence=split_sentence, strategy=strategy)

    model = HybridPreferenceMatching(item_embeddings_path=item_embeddings_path, id_to_index_path=id_to_index_path,
                                     restaurants=restaurants, id_name_map=id_name_map, save_directory=save_path,
                                     split_sentence=split_sentence, cosine=cosine,
                                     n=rate_size, seed=seed, subsample=subsample)

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
        ev_res['Sentence segmentation is average'] = 'NA'
        ev_res['k_S'] = 'NA'

    eval_result = evaluation_prediction(true_labels, query_score_ranked)
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

    return eval_result, output_message