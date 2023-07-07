import json
import random
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
import os
import uuid
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
import tensorflow as tf
from transformers import BertModel, AutoConfig
from transformers import create_optimizer
from tensorflow import keras
import numpy as np

from Neural_PM.finetune.warmup_weights import load_warmups

from keras.initializers import Constant
from Neural_PM.finetune.models.review_model import PassageModel,cross_replica_concat


class ItemModel(tf.keras.Model):
    def __init__(self, passage_encoder, num_passages_per_question, model_config, id_to_index, strategy=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # self.query_encoder = query_encoder
        # self.item_embeddings = tf.Variable(tf.random.uniform(shape=[50, 768], maxval=1, minval=-1, dtype=tf.float32))
        if model_config['warmup']:
            weights = load_warmups(model_config['warmup_weights'], id_to_index)
            print(weights.shape)
            self.item_embeddings = tf.keras.layers.Embedding(50, 768,
                                                             embeddings_initializer=Constant(weights), trainable=True)
        else:
            self.item_embeddings = tf.keras.layers.Embedding(50, 768,
                                                             embeddings_initializer=tf.keras.initializers.RandomUniform(
                                                                 minval=-1.0, maxval=1.0, seed=None), trainable=True)
        self.passage_encoder = passage_encoder
        self.num_passages_per_question = num_passages_per_question
        self.model_config = model_config
        self.sym_num = 50
        self.tpu = self.model_config['tpu']
        self.id_to_index = id_to_index
        # TODO: add item id map
        self.strategy = strategy
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                                     from_logits=True)

    # def get_restaurant_id_map(self, X):
    #     restaurants = []
    #     for x in X:
    #         if len(restaurants) == 50:
    #             # TODO: make it so we can change restaurants to more than 50
    #             self.item_id_index_map = restaurants
    #             break
    #         if x['passages']['business_id'] not in restaurants:
    #             restaurants.append(x['passages']['business_id'])
    #     self.item_id_index_map = restaurants

    def calculate_loss(self, logits, asym=False):
        num_queries = tf.shape(logits)[0]
        num_candidates = tf.shape(logits)[1]

        if asym:
            labels = tf.convert_to_tensor(
                [i for i in range(0, (self.sym_num * self.num_passages_per_question),
                                  self.num_passages_per_question)])
        else:
            labels = tf.convert_to_tensor(
                [i for i in range(0, (self.model_config['GLOBAL_BATCH_SIZE'] * self.num_passages_per_question),
                                  self.num_passages_per_question)])
        # print('labels Shape', labels.shape)
        # print([i for i in range(0,(GLOBAL_BATCH_SIZE*self.num_passages_per_question),self.num_passages_per_question)])
        loss = self.loss_fn(labels, logits)
        scale_loss = tf.reduce_sum(loss) * (1. / self.model_config['GLOBAL_BATCH_SIZE'])
        return scale_loss

    def passage_forward(self, X):
        input_shape = (self.model_config['batch_size_per_replica'] * self.num_passages_per_question,
                       self.model_config['passage_max_seq_len'])

        input_ids = tf.reshape(X["passage_input_ids"], input_shape)
        attention_mask = tf.reshape(X["passage_attention_mask"], input_shape)
        # token_type_ids = tf.reshape(X["passage_token_type_ids"], input_shape)
        outputs = self.passage_encoder([input_ids, attention_mask], training=True)
        return outputs

    def query_forward(self, X):
        # input_shape = (self.model_config['batch_size_per_replica'], self.model_config['query_max_seq_len'])
        outputs = self.item_embeddings(X['restaurants'], training=True)
        return outputs

    def get_embeddings(self):
        inp = tf.convert_to_tensor(np.array(list(range(50))))
        return self.item_embeddings(inp).numpy()

    def get_loss(self, X):
        passage_embeddings = self.passage_forward(X)
        query_embeddings = self.query_forward(X)
        # TODO: define a weight matrix of (50,768) and take slice of (global_bs,768) as query_embeddings
        # TODO: and define a call back to save that weight matrix
        if self.tpu:
            global_passage_embeddings = cross_replica_concat(passage_embeddings, 32)
            global_query_embeddings = cross_replica_concat(query_embeddings, 16)
        else:
            global_passage_embeddings = passage_embeddings
            global_query_embeddings = query_embeddings

        if self.model_config['asym_negative']:
            asym_rev = global_query_embeddings[self.sym_num:, :]

            global_query_embeddings = global_query_embeddings[:self.sym_num, :]
            global_passage_embeddings = tf.concat([global_passage_embeddings, asym_rev], axis=0)
        similarity_scores = tf.linalg.matmul(global_query_embeddings, global_passage_embeddings, transpose_b=True)
        if self.model_config['reverse_item_embedding']:

            loss = self.calculate_loss(tf.transpose(similarity_scores))
        else:
            loss = self.calculate_loss(similarity_scores, self.model_config['asym_negative'])

        if self.tpu:
            loss = loss / self.strategy.num_replicas_in_sync
        return loss

    def train_step(self, X):
        with tf.GradientTape() as tape:
            loss = self.get_loss(X)

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, X):
        loss = self.get_loss(X)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def get_item_model(finetune_setting, data_config, strategy, one_epoch_steps):
    passage_encoder = PassageModel(finetune_setting)
    item_model = ItemModel(passage_encoder,
                           num_passages_per_question=data_config.num_positives + data_config.num_hard_negatives,
                           model_config=finetune_setting, strategy=strategy,
                           id_to_index=finetune_setting['id_to_index'])
    optimizer, lr_schedule = create_optimizer(init_lr=finetune_setting['learning_rate'],
                                              num_train_steps=finetune_setting['epochs'] * one_epoch_steps,
                                              num_warmup_steps=(finetune_setting['epochs'] * one_epoch_steps) * 0.1)
    item_model.compile(optimizer=optimizer)
    return item_model, optimizer, lr_schedule
