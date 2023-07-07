# Mahdi Abdollahpour, 16/03/2022, 10:51 AM, PyCharm, Neural_PM


from transformers import AutoConfig, AutoTokenizer, TFAutoModel, AutoModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# import tensorflow_addons
# import tensorflow_hub as hub
# Progress bar
from tqdm import tqdm
# import torch


# https://github.com/huggingface/transformers/issues/1950

def create_model(BERT_name, from_pt=False):
    ## BERT encoder
    encoder = TFAutoModel.from_pretrained(BERT_name, from_pt=from_pt)

    ## Model
    input_ids = layers.Input(shape=(None,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(None,), dtype=tf.int32)
    # token_type_ids = layers.Input(shape=(None,), dtype=tf.int32)

    embedding = encoder(
        # input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        input_ids=input_ids, attention_mask=attention_mask
    )

    model = keras.Model(
        # inputs=[input_ids, attention_mask, token_type_ids],
        inputs=[input_ids, attention_mask],
        outputs=embedding, )

    model.compile()
    return model, input_ids.name, attention_mask.name


class BERT_model:
    def __init__(self, BERT_name, tokenizer_name, from_pt=False):
        """

        :param BERT_name: name or address of language prefernce_matching
        :param tokenizer_name: name or address of the tokenizer
        """
        self.BERT_name = BERT_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.bert_model, self.name1, self.name2 = create_model(BERT_name, from_pt)
        # self.bert_model = TFAutoModel.from_pretrained(pre_trained_model_name)
        print(BERT_name)

        # if torch.cuda.is_available():
        #   print("GPU available")
        #   self.bert_model.cuda()
        # self.bert_model.eval()

    def apply_batch(self, texts, strategy=None, bs=48, verbose=0):
        # if strategy is not None:
        #     with strategy.scope():
        #         self.bert_model = create_model("bert-base-uncased")
        tokenized_review = self.tokenizer.batch_encode_plus(
            texts,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
            # truncation_strategy='longest_first',
            padding="max_length",
            return_token_type_ids=True,
        )

        data = {self.name1: tokenized_review['input_ids'],
                self.name2: tokenized_review['attention_mask'],
                # 'input_3': tokenized_review['token_type_ids']
                }
        # print(np.array(tokenized_review['input_ids']).shape)
        if strategy is not None:
            with strategy.scope():
                dataset = tf.data.Dataset.from_tensor_slices(data).batch(bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
                outputs = self.bert_model.predict(dataset, verbose=verbose)
                # print(outputs['last_hidden_state'].shape)
                return outputs['last_hidden_state'][:, 0, :].reshape(-1, 768)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE).batch(bs, drop_remainder=False)
            outputs = self.bert_model.predict(dataset, verbose=verbose)
            # print(outputs['last_hidden_state'].shape)
            return outputs['last_hidden_state'][:, 0, :].reshape(-1, 768)

    def apply(self, text, strategy=None):
        """

        :param text: str
        :return: language prefernce_matching representation of the text
        """
        tokenized_review = self.tokenizer(text)
        if strategy is not None:
            # print('Using tpu')
            with strategy.scope():
                outputs = self.bert_model.predict(tokenized_review)
        else:
            outputs = self.bert_model.predict([np.array(tokenized_review['input_ids']).reshape(1, -1),
                                               np.array(tokenized_review['attention_mask']).reshape(1, -1),
                                               np.array(tokenized_review['token_type_ids']).reshape(1, -1)])

        cls_vec = outputs[0][0, 0, :]
        # results={}
        # query_input = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        # with torch.no_grad():
        #     if (query_input["input_ids"].shape[-1]):
        #         query_input["input_ids"] = query_input["input_ids"][:, :512]
        #     if (query_input["attention_mask"].shape[-1]):
        #         query_input["attention_mask"] = query_input["attention_mask"][:, :512]
        #     if torch.cuda.is_available():
        #         query_input = query_input.to(device='cuda')
        #     bert_out = self.bert_model(**query_input)
        #     # print(bert_out.keys())
        #     # print('1',bert_out[0].shape,bert_out[1].shape)
        #     # print('2',bert_out[0][:,0,:].shape)
        #     # print(bert_out[1].shape)
        #     # print('3',len(bert_out))
        #     embedding = bert_out[0][:, 0, :].squeeze(0)
        # return embedding
        # print(cls_vec.shape)
        # print(outputs[0].shape)
        return cls_vec.reshape(1, -1)
