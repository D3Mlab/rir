from sklearn.utils import shuffle

from Neural_PM.finetune.train import *
from Neural_PM.prefernce_matching.PM_experiment import *
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
from Neural_PM.finetune.sampling.standard_dicts import *


def get_positive_samples(finetune_setting, val_ratio=0.2, seed=100):
    if finetune_setting['item_embedding']:
        print('Item Embedding Data Sampling...')
        train_reviews, val_reviews, all_reviews = get_train_val_dfs(finetune_setting["train_data_path"],
                                                                    finetune_setting)
        restaurants = all_reviews.business_id.unique()
        # cats = all_reviews.cat.unique()
        all_positive_samples_train, all_positive_samples_val = get_positive_samples_item_embedding(
            train_reviews, val_reviews, restaurants)
        dicts_train = positive_sub_sampler(all_positive_samples_train, finetune_setting['positive_pair_per_restaurant'],
                                           seed=seed)
        dicts_val = positive_sub_sampler(all_positive_samples_val,
                                         int(finetune_setting['positive_pair_per_restaurant'] * val_ratio), seed=seed)
        dicts_standard_train = read_review_dicts_item(dicts_train)
        dicts_standard_val = read_review_dicts_item(dicts_val)

    elif not finetune_setting['ir_style']:
        additional_df = None

        if finetune_setting['sym_negative']:
            additional_df = get_sym_reviews(finetune_setting['all_data'], finetune_setting['train_data_path'],
                                            number_of_restaurants=finetune_setting['sym_negative_num'])

        train_reviews, val_reviews, all_reviews = get_train_val_dfs(finetune_setting["train_data_path"],
                                                                    finetune_setting, additional_df=additional_df)
        restaurants = all_reviews.business_id.unique()
        # cats = all_reviews.cat.unique()

        if finetune_setting['high_tfidf']:
            all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews = get_positive_samples_contrastive_with_high_tfidf(
                train_reviews, val_reviews, restaurants, tfidf_file=finetune_setting['tfidf_scores_file'],
                threshold=finetune_setting['tfidf_threshold'])
        elif finetune_setting['same_cat']:
            assert cat is not None,'cat is None'
            all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews = get_positive_samples_contrastive_same_category(
                train_reviews, val_reviews, cats)
        elif finetune_setting['same_cluster'] or finetune_setting['diff_cluster']:
            # cr = ClusterReviews(finetune_setting["train_data_path"],BERTVectorizerModel(strategy),finetune_setting["n_clusters"])
            # print('Clustering...')
            # cr.cluster()
            # print('Savin Clustering Results...')
            # cr.save()
            cluster_setting = {"model_name": finetune_setting['model_name'], "tpu": finetune_setting['tpu'],
                               "train_data_path": finetune_setting["train_data_path"],
                               'true_labels_path': finetune_setting['true_labels_path'],
                               'filtered_review_data': finetune_setting['filtered_review_data'],
                               'n_clusters': finetune_setting['n_clusters'],
                               'tfidf_feature': finetune_setting['tfidf_feature'],
                               'embedded_reviews': finetune_setting['embedded_reviews_path'],
                               'new_data_path': 'data_with_cluster_label_' + str(
                                   finetune_setting['n_clusters']) + '_' + str(
                                   finetune_setting['number']) + '.csv',
                               }
            with open(os.path.join(finetune_setting['save_path'], 'cluster_setting.json'), 'wb') as f:
                f.write(str.encode(str(cluster_setting)))

            run_clustering(cluster_setting)
            all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews = get_positive_samples_contrastive_same_cluster(
                train_reviews, val_reviews, restaurants)
            if finetune_setting['same_cluster']:
                print('Same cluster label filter...')
                all_positive_samples_train = same_field_filter(all_positive_samples_train, 2)
                all_positive_samples_val = same_field_filter(all_positive_samples_val, 2)
            else:
                print('Diff cluster label filter...')
                all_positive_samples_train = diff_field_filter(all_positive_samples_train, 2)
                all_positive_samples_val = diff_field_filter(all_positive_samples_val, 2)

        else:

            all_positive_samples_train, all_positive_samples_val, train_reviews, val_reviews = get_positive_samples_contrastive(
                train_reviews, val_reviews, restaurants)

        if finetune_setting['least_similar'] or finetune_setting['most_similar'] or finetune_setting['hard_negative']:
            print('Loading Embeddings from ', finetune_setting['embedded_reviews_path'])
            Xfile = open(finetune_setting['embedded_reviews_path'], 'rb')
            X = pickle.load(Xfile)
            Xfile.close()
            sims = np.dot(X, X.T)
            print('sims shape', sims.shape)
            del X

        if finetune_setting['same_rating']:
            print('same rating filter...')
            all_positive_samples_train = same_field_filter(all_positive_samples_train, 2)
            all_positive_samples_val = same_field_filter(all_positive_samples_val, 2)
        if finetune_setting['diff_rating']:
            print('diff rating filter...')
            all_positive_samples_train = different_field_filter(all_positive_samples_train, 2,
                                                                finetune_setting['rate_diff'])
            all_positive_samples_val = different_field_filter(all_positive_samples_val, 2,
                                                              finetune_setting['rate_diff'])
        # we subsample so it would be balanced
        if finetune_setting['least_similar'] or finetune_setting['most_similar']:
            if finetune_setting['most_similar']:
                print('most_similar subsampling...')
            else:
                print('least_similar subsampling...')

            reverse = finetune_setting['most_similar']
            dicts_train = positive_sub_sampler_on_similarity(all_positive_samples_train,
                                                             sims,
                                                             finetune_setting['positive_pair_per_restaurant'],
                                                             seed=seed, reverse=reverse,
                                                             start=finetune_setting['percentile'])
            dicts_val = positive_sub_sampler_on_similarity(all_positive_samples_val,
                                                           sims,
                                                           int(finetune_setting[
                                                                   'positive_pair_per_restaurant'] * val_ratio),
                                                           seed=seed, reverse=reverse,
                                                           start=finetune_setting['percentile'])
        else:  # random
            dicts_train = positive_sub_sampler(all_positive_samples_train,
                                               finetune_setting['positive_pair_per_restaurant'],
                                               seed=seed)
            dicts_val = positive_sub_sampler(all_positive_samples_val,
                                             int(finetune_setting['positive_pair_per_restaurant'] * val_ratio),
                                             seed=seed)
        if finetune_setting['hard_negative']:
            dicts_train = get_hard_negatives(dicts_train, train_reviews, sims,
                                             finetune_setting['hard_negative_num'])
            dicts_val = get_hard_negatives(dicts_val, train_reviews, sims,
                                           finetune_setting['hard_negative_num'])

        if finetune_setting['asym_negative']:
            asym_reviews = get_asym_reviews(finetune_setting['all_data'], finetune_setting['train_data_path'])
            asym_reviews = shuffle(asym_reviews, random_state=seed)
            asym_needed = len(dicts_train) * finetune_setting['positive_pair_per_restaurant']
            asym_needed_val = len(dicts_val) * int(finetune_setting['positive_pair_per_restaurant'] * val_ratio)
            assert asym_needed + asym_needed_val <= len(asym_reviews), "Error: Not enough asym reviews provided!"
            asym_reviews_train = asym_reviews[:asym_needed]
            asym_reviews_val = asym_reviews[asym_needed:asym_needed + asym_needed_val]
        else:
            asym_reviews = None
            asym_reviews_train = None
            asym_reviews_val = None

        if finetune_setting['asym_negative']:
            dicts_standard_train = read_review_dicts_with_asym(dicts_train, asym_reviews_train,
                                                               finetune_setting["asym_num"])
            dicts_standard_val = read_review_dicts_with_asym(dicts_val, asym_reviews_val,
                                                             finetune_setting["asym_num"])
        else:
            dicts_standard_train = read_review_dicts(dicts_train, with_hard_negatives=finetune_setting['hard_negative'],
                                                     prepend=finetune_setting['prepend_categories'],
                                                     prepend_both=finetune_setting['prepend_both'],
                                                     subsample_query=finetune_setting['subsample_query'],
                                                   subsample_query_sentence=finetune_setting['subsample_query_sentence'])
            dicts_standard_val = read_review_dicts(dicts_val, with_hard_negatives=finetune_setting['hard_negative'],
                                                   prepend=finetune_setting['prepend_categories'],
                                                   prepend_both=finetune_setting['prepend_both'],
                                                   subsample_query=finetune_setting['subsample_query'],
                                                   subsample_query_sentence=finetune_setting['subsample_query_sentence'])
    else:
        additional_df = None
        train_reviews, val_reviews, all_reviews = get_train_val_dfs_neural_ir(finetune_setting["train_data_path"],
                                                                              finetune_setting,
                                                                              additional_df=additional_df,
                                                                              type_sampling=finetune_setting[
                                                                                  "ir_sampling_type"], seed=seed)
        restaurants = all_reviews.business_id.unique()
        # cats = all_reviews.cat.unique()

        all_positive_samples_train, all_positive_samples_val = get_positive_samples_neural_ir(train_reviews,
                                                                                              val_reviews, restaurants)
        # we don't subsample for ir style
        dicts_standard_train = read_review_dicts_ir_style(all_positive_samples_train)
        dicts_standard_val = read_review_dicts_ir_style(all_positive_samples_val)
        dicts_standard_train = shuffle(dicts_standard_train, random_state=seed)
        dicts_standard_val = shuffle(dicts_standard_val, random_state=seed)
    print('dicts_standard_train len', len(dicts_standard_train))
    print('dicts_standard_val len', len(dicts_standard_val))
    return dicts_standard_train, dicts_standard_val, restaurants
