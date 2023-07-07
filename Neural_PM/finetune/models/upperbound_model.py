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
from Neural_PM.finetune.models.review_model import PassageModel,BiEncoderModel,cross_replica_concat
from keras.initializers import Constant

class BiEncoderModelUpperBound(BiEncoderModel):
    def calculate_loss(self, logits, asym=False, temperature=1.0):
        logits = logits / temperature

        labels = tf.convert_to_tensor(
            [0 for i in range(0, (self.model_config['GLOBAL_BATCH_SIZE'] * self.num_passages_per_question),
                              self.num_passages_per_question)])
        # print('labels Shape', labels.shape)
        # print([i for i in range(0,(GLOBAL_BATCH_SIZE*self.num_passages_per_question),self.num_passages_per_question)])
        loss = self.loss_fn(labels, logits)
        scale_loss = tf.reduce_sum(loss) * (1. / self.model_config['GLOBAL_BATCH_SIZE'])
        return scale_loss

    def get_loss(self, X):
        passage_embeddings = self.passage_forward(X)

        hn_embeddings = self.hardnegatives_forward(X)
        query_embeddings = self.query_forward(X)

        if self.tpu:
            global_passage_embeddings = cross_replica_concat(passage_embeddings, 32)
            global_hn_embeddings = cross_replica_concat(hn_embeddings, 32)
            global_query_embeddings = cross_replica_concat(query_embeddings, 16)
        else:
            global_passage_embeddings = passage_embeddings
            global_hn_embeddings = hn_embeddings
            global_query_embeddings = query_embeddings

        mat = tf.linalg.matmul(
            global_query_embeddings, global_passage_embeddings, transpose_b=True
        )
        mat = tf.reshape(mat, (1, self.model_config['GLOBAL_BATCH_SIZE'],
                               self.model_config['GLOBAL_BATCH_SIZE']))
        similarity_scores = tf.linalg.diag_part(mat)
        similarity_scores = tf.reshape(similarity_scores, (self.model_config['GLOBAL_BATCH_SIZE'], 1))

        mat2 = tf.linalg.matmul(
            global_query_embeddings, global_hn_embeddings, transpose_b=True
        )
        mat2 = tf.reshape(mat2, (self.model_config['hard_negative_num'], self.model_config['GLOBAL_BATCH_SIZE'],
                                 self.model_config['GLOBAL_BATCH_SIZE']))
        hn_scores = tf.linalg.diag_part(mat2)
        hn_scores = tf.reshape(hn_scores,
                               (self.model_config['GLOBAL_BATCH_SIZE'], self.model_config['hard_negative_num']))
        final_sim = tf.concat([similarity_scores, hn_scores], axis=1)

        loss = self.calculate_loss(final_sim, self.model_config['asym_negative'],
                                   self.model_config['temperature'])

        if self.tpu:
            loss = loss / self.strategy.num_replicas_in_sync
        return loss


def get_upperbound_bimodel(finetune_setting, data_config, strategy, one_epoch_steps):
    passage_encoder = PassageModel(finetune_setting)
    bi_model = BiEncoderModelUpperBound(passage_encoder,
                                        num_passages_per_question=data_config.num_positives,
                                        model_config=finetune_setting, strategy=strategy)
    optimizer, lr_schedule = create_optimizer(init_lr=finetune_setting['learning_rate'],
                                              num_train_steps=finetune_setting['epochs'] * one_epoch_steps,
                                              num_warmup_steps=(finetune_setting['epochs'] * one_epoch_steps) * 0.1)
    bi_model.compile(optimizer=optimizer)
    return bi_model, optimizer, lr_schedule


