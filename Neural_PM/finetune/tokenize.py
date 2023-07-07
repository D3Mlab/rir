# Mahdi Abdollahpour, 08/04/2022, 12:23 AM, PyCharm, lgeconvrec

from tqdm import tqdm
import numpy as np


def encode_query_passage(tokenizer, dicts, finetune_setting, data_config):
    passage_input_ids = []
    hn_input_ids = []
    passage_token_type_ids = []
    hn_token_type_ids = []
    passage_attention_mask = []
    hn_attention_mask = []

    queries = []
    for i in tqdm(range(len(dicts))):
        di = dicts[i]
        di_query = di['query']

        # TODO: mke business ID ,
        di_passages = di['passages']
        di_positives = [pi['text'] for pi in di_passages if pi['label'] == "positive"]
        di_negatives = [ni['text'] for ni in di_passages if ni['label'] == "hard_negative"]

        if data_config.num_positives == len(di_positives) and data_config.num_hard_negatives == len(di_negatives):
            queries.append(di_query)  # rev1
            i_passages = di_positives  # rev2
            i_passage_inputs = tokenizer.batch_encode_plus(
                i_passages,
                max_length=finetune_setting['passage_max_seq_len'],
                add_special_tokens=True,
                truncation=True,
                truncation_strategy='longest_first',
                padding="max_length",
                return_token_type_ids=True,
            )
            # print("i_passage_inputs['input_ids'] shape",np.array(i_passage_inputs['input_ids']).shape)
            passage_input_ids.append(np.array(i_passage_inputs['input_ids']))
            passage_token_type_ids.append(np.array(i_passage_inputs['token_type_ids']))
            passage_attention_mask.append(np.array(i_passage_inputs['attention_mask']))
            if finetune_setting['hard_negative']:
                i_hn_inputs = tokenizer.batch_encode_plus(
                    di_negatives,
                    max_length=finetune_setting['passage_max_seq_len'],
                    add_special_tokens=True,
                    truncation=True,
                    truncation_strategy='longest_first',
                    padding="max_length",
                    return_token_type_ids=True,
                )
                # print("i_passage_inputs['input_ids'] shape",np.array(i_passage_inputs['input_ids']).shape)
                hn_input_ids.append(np.array(i_hn_inputs['input_ids']))
                hn_token_type_ids.append(np.array(i_hn_inputs['token_type_ids']))
                hn_attention_mask.append(np.array(i_hn_inputs['attention_mask']))

    print("len queries:", len(queries))
    query_inputs = tokenizer.batch_encode_plus(
        queries,
        max_length=finetune_setting['query_max_seq_len'],
        add_special_tokens=True,
        truncation=True,
        truncation_strategy='longest_first',
        padding="max_length",
        return_token_type_ids=True,
        return_tensors="np"
    )

    return_dict = {
        "query_input_ids": query_inputs['input_ids'],
        "query_token_type_ids": query_inputs['token_type_ids'],
        "query_attention_mask": query_inputs['attention_mask'],
        "passage_input_ids": np.array(passage_input_ids),
        "passage_token_type_ids": np.array(passage_token_type_ids),
        "passage_attention_mask": np.array(passage_attention_mask),

    }
    if finetune_setting['hard_negative']:
        return_dict["hn_input_ids"] = np.array(hn_input_ids)
        return_dict["hn_token_type_ids"] = np.array(hn_token_type_ids)
        return_dict["hn_attention_mask"] = np.array(hn_attention_mask)
    # if finetune_setting['asym_negative']:
    #     asym_inputs = tokenizer.batch_encode_plus(
    #         asym_reviews,
    #         max_length=finetune_setting['passage_max_seq_len'],
    #         add_special_tokens=True,
    #         truncation=True,
    #         truncation_strategy='longest_first',
    #         padding="max_length",
    #         return_token_type_ids=True,
    #         return_tensors="np"
    #     )
    #     return_dict["query_input_ids"] = asym_inputs['input_ids']
    #     return_dict["query_token_type_ids"] = asym_inputs['token_type_ids']
    #     return_dict["query_attention_mask"] = asym_inputs['attention_mask']

    return return_dict

# TODO: add item id to query and over write it, only returning one list not tokenizing


def encode_item_passage(tokenizer, dicts, finetune_setting, data_config):
    passage_input_ids = []
    passage_token_type_ids = []
    passage_attention_mask = []
    restaurants = []
    # queries = []
    id_to_index = finetune_setting['id_to_index']
    for i in tqdm(range(len(dicts))):
        di = dicts[i]
        di_query = di['query']
        di_item = di['business_id']
        # TODO: mke business ID ,
        di_passages = di['passages']
        di_positives = [pi['text'] for pi in di_passages if pi['label'] == "positive"]
        di_negatives = [ni['text'] for ni in di_passages if ni['label'] == "hard_negative"]
        # How are we ensuring the correctness of the below line?
        if data_config.num_positives == len(di_positives) and data_config.num_hard_negatives == len(di_negatives):
            restaurants.append(id_to_index[di_item])
            # queries.append(di_query)  # rev1
            i_passages = di_positives + di_negatives  # rev2
            i_passage_inputs = tokenizer.batch_encode_plus(
                i_passages,
                max_length=finetune_setting['passage_max_seq_len'],
                add_special_tokens=True,
                truncation=True,
                truncation_strategy='longest_first',
                padding="max_length",
                return_token_type_ids=True,
            )
            # print("i_passage_inputs['input_ids'] shape",np.array(i_passage_inputs['input_ids']).shape)
            passage_input_ids.append(np.array(i_passage_inputs['input_ids']))
            passage_token_type_ids.append(np.array(i_passage_inputs['token_type_ids']))
            passage_attention_mask.append(np.array(i_passage_inputs['attention_mask']))

    # print("len queries:", len(queries))
    # query_inputs = tokenizer.batch_encode_plus(
    #     queries,
    #     max_length=finetune_setting['query_max_seq_len'],
    #     add_special_tokens=True,
    #     truncation=True,
    #     truncation_strategy='longest_first',
    #     padding="max_length",
    #     return_token_type_ids=True,
    #     return_tensors="np"
    # )

    return_dict = {
        # "query_input_ids": query_inputs['input_ids'],
        # "query_token_type_ids": query_inputs['token_type_ids'],
        # "query_attention_mask": query_inputs['attention_mask'],
        "restaurants": restaurants,
        "passage_input_ids": np.array(passage_input_ids),
        "passage_token_type_ids": np.array(passage_token_type_ids),
        "passage_attention_mask": np.array(passage_attention_mask)
    }


    return return_dict