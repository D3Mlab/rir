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


def run_experiments_with_item_embedding(item_embeddings_path, id_to_index_path, filtered_review_data,
                                        create_matrices=True,
                                        BERT_name='MSMARCO',
                                        save_path='matrices/',
                                        save_metrics=True, cosine=False, seed=None,
                                        results_save_path='./results/generated_result/',
                                        true_labels_path='./data/CRS dataset-Restaurantwith sample reviews - Binary Data.csv',
                                        strategy=None,
                                        from_pt=False, settings_to_include=None, prepend_categories=False):
    sim = 'dot'
    if cosine:
        sim = 'cosine'
    stripped_BN = BERT_name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')
    stripped_item_embeddings_path = item_embeddings_path.replace('./', '#').replace('/', '#').replace('-', '#').replace(
        '_', '$')
    EXP_NAME = "_".join([stripped_BN, str(stripped_item_embeddings_path), sim, str(seed)])

    df = pd.read_csv(filtered_review_data)
    restaurants = df.groupby(['name'])
    print(df.shape)
    if prepend_categories:
        df['review_text'] = df['categories'] + ' ' + df['review_text']
    true_labels = build_true_labels(true_labels_path)
    queries, restaurants = get_queries_and_resturants(true_labels_path)
    restaurants = restaurants.tolist()

    reviews, id_name_map = sort_reviews_by_business(df, restaurants)
    print(len(reviews))

    if create_matrices or not os.path.exists(save_path):
        BERT_model_name = BERT_MODELS[BERT_name]
        # BERT_model_name = BERT_name
        tokenizer_name = TOEKNIZER_MODELS[BERT_name]
        # tokenizer_name = BERT_name
        if strategy is not None:
            with strategy.scope():
                embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)
        else:
            embedder = BERT_model(BERT_model_name, tokenizer_name, from_pt)

        save_query_matrices(queries, save_path, embedder, strategy=strategy)

    model = ItemPreferenceMatching(item_embeddings_path=item_embeddings_path, id_to_index_path=id_to_index_path,
                                   restaurants=restaurants, id_name_map=id_name_map, save_directory=save_path,
                                   cosine=cosine,
                                   seed=seed)

    outputs, _ = model.predict()

    query_score = outputs

    query_score_ranked = ranking(query_score)

    name_to_id = {v: k for k, v in id_name_map.items()}

    # output_message = create_output_text(queries, best_matching_reviews, name_to_id, restaurants, query_score_ranked,
    #                                     true_labels)

    ev_res = {}
    ev_res['similarity'] = sim
    ev_res['BERT prefernce_matching'] = BERT_name

    eval_result = evaluation_prediction(true_labels, query_score_ranked)
    for key in eval_result:
        ev_res[key] = eval_result[key]
    if settings_to_include is not None:
        for key in settings_to_include:
            ev_res[key] = str(settings_to_include[key])

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    # if save_output:
    #     with open(os.path.join(results_save_path, EXP_NAME + '_output.txt'), mode='w', encoding='utf8') as f:
    #         f.write(output_message)
    if save_metrics:
        with open(os.path.join(results_save_path, EXP_NAME + '_metrics.txt'), mode='w', encoding='utf8') as f:
            f.write(str(eval_result))

    return eval_result, ''



