# Mahdi Abdollahpour, 08/04/2022, 12:23 AM, PyCharm, lgeconvrec

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


# from tqdm import tqdm_notebook

class PassageModel(tf.keras.Model):
    def __init__(self, finetune_setting, **kwargs):
        super().__init__(**kwargs)

        # configuration = AutoConfig.from_pretrained(finetune_setting['model_name'])
        # configuration.hidden_dropout_prob =0
        # configuration.attention_probs_dropout_prob = 0
        self.passage_encoder = TFAutoModel.from_pretrained(finetune_setting['model_name'], from_pt=True
                                                           # ,config=configuration
                                                           )
        self.use_dropout = finetune_setting['dropout'] > 0
        if self.use_dropout:
            self.dropout = tf.keras.layers.Dropout(finetune_setting['dropout'], seed=100)

    def call(self, inputs, training=False, **kwargs):
        # pooled_output = self.passage_encoder(inputs, training=training, **kwargs)[1]
        cls = self.passage_encoder(inputs, training=training, **kwargs)[0][:, 0, :]
        if self.use_dropout:
            cls = self.dropout(cls, training=training)
        # print("P pooled_output :", pooled_output.shape)
        return cls


def cross_replica_concat(values, v_shape):
    context = tf.distribute.get_replica_context()
    gathered = context.all_gather(values, axis=0)

    return tf.roll(
        gathered,
        -context.replica_id_in_sync_group * values.shape[0],  # v_shape,#values.shape[0],
        axis=0
    )


class BiEncoderModel(tf.keras.Model):
    def __init__(self, passage_encoder, num_passages_per_question, model_config, strategy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.num_passages_per_question = num_passages_per_question
        self.model_config = model_config
        self.sym_num = 50
        self.tpu = self.model_config['tpu']
        self.strategy = strategy
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                                     from_logits=True)
        self.add_noise = self.model_config['add_noise']
        self.stddev = self.model_config['stddev']

    def calculate_loss(self, logits, asym=False, temperature=1.0):
        num_queries = tf.shape(logits)[0]
        num_candidates = tf.shape(logits)[1]
        logits = logits / temperature
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

        if self.add_noise:
            noise = tf.random.normal(shape=logits.get_shape(), mean=0.0, stddev=self.stddev, dtype=tf.float32)
            logits += noise
        loss = self.loss_fn(labels, logits)
        scale_loss = tf.reduce_sum(loss) * (1. / self.model_config['GLOBAL_BATCH_SIZE'])
        return scale_loss

    def passage_forward(self, X):
        input_shape = (self.model_config['batch_size_per_replica'] * self.num_passages_per_question,
                       self.model_config['passage_max_seq_len'])

        input_ids = tf.reshape(X["passage_input_ids"], input_shape)
        attention_mask = tf.reshape(X["passage_attention_mask"], input_shape)
        # token_type_ids = tf.reshape(X["passage_token_type_ids"], input_shape)
        # outputs = self.passage_encoder([input_ids, attention_mask, token_type_ids], training=True)
        outputs = self.passage_encoder([input_ids, attention_mask], training=True)
        return outputs

    def query_forward(self, X):
        input_shape = (self.model_config['batch_size_per_replica'], self.model_config['query_max_seq_len'])
        input_ids = tf.reshape(X["query_input_ids"], input_shape)
        attention_mask = tf.reshape(X["query_attention_mask"], input_shape)
        # token_type_ids = tf.reshape(X["query_token_type_ids"], input_shape)
        # outputs = self.passage_encoder([input_ids, attention_mask, token_type_ids], training=True)
        outputs = self.passage_encoder([input_ids, attention_mask], training=True)
        return outputs

    def hardnegatives_forward(self, X):
        input_shape = (self.model_config['batch_size_per_replica'] * self.model_config['hard_negative_num'],
                       self.model_config['passage_max_seq_len'])

        input_ids = tf.reshape(X["hn_input_ids"], input_shape)
        attention_mask = tf.reshape(X["hn_attention_mask"], input_shape)
        outputs = self.passage_encoder([input_ids, attention_mask], training=True)
        return outputs

    def get_loss(self, X):
        passage_embeddings = self.passage_forward(X)
        if self.model_config['hard_negative']:
            hn_embeddings = self.hardnegatives_forward(X)
        query_embeddings = self.query_forward(X)

        if self.tpu:
            global_passage_embeddings = cross_replica_concat(passage_embeddings, 32)
            if self.model_config['hard_negative']:
                global_hn_embeddings = cross_replica_concat(hn_embeddings, 32)
            global_query_embeddings = cross_replica_concat(query_embeddings, 16)
        else:
            global_passage_embeddings = passage_embeddings
            if self.model_config['hard_negative']:
                global_hn_embeddings = hn_embeddings
            global_query_embeddings = query_embeddings

        if self.model_config['asym_negative']:
            asym_rev = global_query_embeddings[self.sym_num:, :]

            global_query_embeddings = global_query_embeddings[:self.sym_num, :]
            global_passage_embeddings = tf.concat([global_passage_embeddings, asym_rev], axis=0)
        similarity_scores = tf.linalg.matmul(global_query_embeddings, global_passage_embeddings, transpose_b=True)

        if self.model_config['hard_negative']:
            # print('global_hn_embeddings', global_hn_embeddings.shape)
            mat = tf.linalg.matmul(
                global_query_embeddings, global_hn_embeddings, transpose_b=True
            )
            mat = tf.reshape(mat, (self.model_config['hard_negative_num'], self.model_config['GLOBAL_BATCH_SIZE'],
                                   self.model_config['GLOBAL_BATCH_SIZE']))

            # print('mat', mat.shape)

            hn_scores = tf.linalg.diag_part(mat)
            # print('hn_scores', hn_scores.shape)
            # print('similarity_scores', similarity_scores.shape)
            hn_scores = tf.reshape(hn_scores,
                                   (self.model_config['GLOBAL_BATCH_SIZE'], self.model_config['hard_negative_num']))
            # print('reshaped hn_scores', hn_scores.shape)
            final_sim = tf.concat([similarity_scores, hn_scores], axis=1)
            # print('final_sim', final_sim.shape)
        else:
            final_sim = similarity_scores
        loss = self.calculate_loss(final_sim, self.model_config['asym_negative'],
                                   self.model_config['temperature'])

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


def get_bimodel(finetune_setting, data_config, strategy, one_epoch_steps):
    passage_encoder = PassageModel(finetune_setting)
    bi_model = BiEncoderModel(passage_encoder,
                              num_passages_per_question=data_config.num_positives,
                              model_config=finetune_setting, strategy=strategy)
    optimizer, lr_schedule = create_optimizer(init_lr=finetune_setting['learning_rate'],
                                              num_train_steps=finetune_setting['epochs'] * one_epoch_steps,
                                              num_warmup_steps=(finetune_setting['epochs'] * one_epoch_steps) * 0.1)
    bi_model.compile(optimizer=optimizer)
    return bi_model, optimizer, lr_schedule
