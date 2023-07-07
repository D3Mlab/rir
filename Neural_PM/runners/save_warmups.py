shorten_name = {
    'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco': 'TASB',
    'bert-base-uncased': 'BERT',
}

import argparse

parser = argparse.ArgumentParser(description='Train LM')

parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--item_name', type=str, default="item_embeddings")
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--true_labels_path', type=str,
                    default='./data/new_binary.csv')
parser.add_argument('--filtered_review_data', type=str,
                    default='./data/50_restaurants_all_rates.csv')
args = parser.parse_args()

from Neural_PM.finetune.warmup_weights import save_warmups

save_warmups(args.model_name, args.filtered_review_data, args.true_labels_path,
             './warmup_creation_matrices_' + shorten_name[args.model_name],
             args.item_name + '_' + shorten_name[args.model_name])
