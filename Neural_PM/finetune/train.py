# Mahdi Abdollahpour, 08/04/2022, 12:28 AM, PyCharm,


import json
import random
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
import os
import uuid
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
import tensorflow as tf
from Neural_PM.finetune.tokenize import encode_query_passage, encode_item_passage
from Neural_PM.finetune.models.review_model import *
from Neural_PM.finetune.tokenize import encode_query_passage
from Neural_PM.finetune.models.item_model import *
from Neural_PM.finetune.models.upperbound_model import *
from transformers import create_optimizer
from Neural_PM.finetune.callbacks import SaveLM, SaveLMAndItemEmbedding
from tensorflow import keras
from keras.callbacks import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt


# from tqdm import tqdm_notebook


def review_finetune(dicts_train, dicts_val, finetune_setting, strategy=None):
    nh = 0
    if finetune_setting['hard_negative']:
        nh = finetune_setting['hard_negative_num']

    class DataConfig:
        num_positives = 1
        num_hard_negatives = nh
        asym_negative = finetune_setting['asym_negative']
        temperature = finetune_setting['temperature']
        hard_negative = finetune_setting['hard_negative']

    data_config = DataConfig()

    tokenizer = AutoTokenizer.from_pretrained(finetune_setting['model_name'])

    tpu = finetune_setting['tpu']
    BATCH_SIZE_PER_REPLICA = finetune_setting['batch_size_per_replica']
    # GLOBAL_asym = 'No_Asym'
    if tpu:
        print('Using TPU')
        # strategy = setup_tpu()
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica'] * strategy.num_replicas_in_sync
    else:
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica']

    N_EPOCHS = finetune_setting['epochs']
    one_epoch_steps = int(len(dicts_train) / GLOBAL_BATCH_SIZE)
    finetune_setting['GLOBAL_BATCH_SIZE'] = GLOBAL_BATCH_SIZE

    print('one_epoch_steps:', one_epoch_steps, 'epochs:', N_EPOCHS, 'BATCH_SIZE_PER_REPLICA:', BATCH_SIZE_PER_REPLICA,
          'GLOBAL_BATCH_SIZE:', GLOBAL_BATCH_SIZE)

    X = encode_query_passage(tokenizer, dicts_train, finetune_setting, data_config)
    X_val = encode_query_passage(tokenizer, dicts_val, finetune_setting, data_config)

    def setup_model_and_data():
        bi_model, optimizer, lr_schedule = get_bimodel(finetune_setting, data_config, strategy, one_epoch_steps)
        train_ds = tf.data.Dataset.from_tensor_slices(X).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        val_ds = tf.data.Dataset.from_tensor_slices(X_val).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        return bi_model, train_ds, val_ds

    if tpu:
        with strategy.scope():
            bi_model, train_ds, val_ds = setup_model_and_data()
    elif finetune_setting['gpu']:
        with tf.device('/device:GPU:0'):
            bi_model, train_ds, val_ds = setup_model_and_data()
    else:
        with tf.device('/cpu:0'):
            bi_model, train_ds, val_ds = setup_model_and_data()

    earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                          patience=finetune_setting['patience'])

    save_path = finetune_setting['save_path']
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    csv_logger = CSVLogger(os.path.join(save_path, "model_history_log.csv"), append=True)
    bi_model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds,
                 callbacks=[SaveLM(save_path), earlyStop_callback, csv_logger])


def item_embedding_finetune(dicts_train, dicts_val, finetune_setting, strategy=None):
    class DataConfig:
        num_positives = 1
        num_hard_negatives = 0
        asym_negative = finetune_setting['asym_negative']

    data_config = DataConfig()

    tokenizer = AutoTokenizer.from_pretrained(finetune_setting['model_name'])

    tpu = finetune_setting['tpu']
    BATCH_SIZE_PER_REPLICA = finetune_setting['batch_size_per_replica']
    # GLOBAL_asym = 'No_Asym'
    if tpu:
        print('Using TPU')
        # strategy = setup_tpu()
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica'] * strategy.num_replicas_in_sync
    else:
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica']

    N_EPOCHS = finetune_setting['epochs']
    one_epoch_steps = int(len(dicts_train) / GLOBAL_BATCH_SIZE)
    finetune_setting['GLOBAL_BATCH_SIZE'] = GLOBAL_BATCH_SIZE

    print('one_epoch_steps:', one_epoch_steps, 'epochs:', N_EPOCHS, 'BATCH_SIZE_PER_REPLICA:', BATCH_SIZE_PER_REPLICA,
          'GLOBAL_BATCH_SIZE:', GLOBAL_BATCH_SIZE)
    # TODO: make restaurant index map from X

    keys = finetune_setting['keys']
    id_to_index = {}
    for i, key in enumerate(list(keys)):
        id_to_index[key] = i
    finetune_setting['id_to_index'] = id_to_index
    X = encode_item_passage(tokenizer, dicts_train, finetune_setting, data_config)
    X_val = encode_item_passage(tokenizer, dicts_val, finetune_setting, data_config)

    def setup_model_and_data():
        # TODO: Declare item embedding model

        bi_model, optimizer, lr_schedule = get_item_model(finetune_setting, data_config, strategy, one_epoch_steps)
        train_ds = tf.data.Dataset.from_tensor_slices(X).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        val_ds = tf.data.Dataset.from_tensor_slices(X_val).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        return bi_model, train_ds, val_ds

    if tpu:
        with strategy.scope():
            bi_model, train_ds, val_ds = setup_model_and_data()
    elif finetune_setting['gpu']:
        with tf.device('/device:GPU:0'):
            bi_model, train_ds, val_ds = setup_model_and_data()
    else:
        bi_model, train_ds, val_ds = setup_model_and_data()

    earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                          patience=finetune_setting['patience'])

    save_path = finetune_setting['save_path']
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    csv_logger = CSVLogger(os.path.join(save_path, "model_history_log.csv"), append=True)
    bi_model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds,
                 callbacks=[SaveLMAndItemEmbedding(save_path), earlyStop_callback, csv_logger])
    save_loss_plot(save_path, "model_history_log.csv")


def save_loss_plot(save_path, file_name):
    df = pd.read_csv(os.path.join(save_path, file_name))
    train_loss = df.loss.values
    val_loss = df.val_loss.values
    x = list(range(len(train_loss)))
    plt.scatter(x, train_loss)
    plt.scatter(x, val_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.ylabel('Loss Value')
    plt.savefig(os.path.join(save_path, "loss_plot.png"))


def upperbound_finetune(dicts_train, dicts_val, finetune_setting, strategy=None):
    nh = 0
    if finetune_setting['hard_negative']:
        nh = finetune_setting['hard_negative_num']

    class DataConfig:
        num_positives = 1
        num_hard_negatives = nh
        asym_negative = finetune_setting['asym_negative']
        temperature = finetune_setting['temperature']
        hard_negative = finetune_setting['hard_negative']

    data_config = DataConfig()

    tokenizer = AutoTokenizer.from_pretrained(finetune_setting['model_name'])

    tpu = finetune_setting['tpu']
    BATCH_SIZE_PER_REPLICA = finetune_setting['batch_size_per_replica']
    # GLOBAL_asym = 'No_Asym'
    if tpu:
        print('Using TPU')
        # strategy = setup_tpu()
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica'] * strategy.num_replicas_in_sync
    else:
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
        # if finetune_setting['asym_negative']:
        # GLOBAL_asym = finetune_setting['asym_per_replica']

    N_EPOCHS = finetune_setting['epochs']
    one_epoch_steps = int(len(dicts_train) / GLOBAL_BATCH_SIZE)
    finetune_setting['GLOBAL_BATCH_SIZE'] = GLOBAL_BATCH_SIZE

    print('UPPERBOUND!, one_epoch_steps:', one_epoch_steps, 'epochs:', N_EPOCHS, 'BATCH_SIZE_PER_REPLICA:',
          BATCH_SIZE_PER_REPLICA,
          'GLOBAL_BATCH_SIZE:', GLOBAL_BATCH_SIZE)

    X = encode_query_passage(tokenizer, dicts_train, finetune_setting, data_config)
    X_val = encode_query_passage(tokenizer, dicts_val, finetune_setting, data_config)

    def setup_model_and_data():
        bi_model, optimizer, lr_schedule = get_upperbound_bimodel(finetune_setting, data_config, strategy,
                                                                  one_epoch_steps)
        train_ds = tf.data.Dataset.from_tensor_slices(X).shuffle(GLOBAL_BATCH_SIZE * 10).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        val_ds = tf.data.Dataset.from_tensor_slices(X_val).shuffle(GLOBAL_BATCH_SIZE * 10).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        return bi_model, train_ds, val_ds

    if tpu:
        with strategy.scope():
            bi_model, train_ds, val_ds = setup_model_and_data()
    elif finetune_setting['gpu']:
        with tf.device('/device:GPU:0'):
            bi_model, train_ds, val_ds = setup_model_and_data()
    else:
        bi_model, train_ds, val_ds = setup_model_and_data()

    earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min',
                                                          patience=finetune_setting['patience'])

    save_path = finetune_setting['save_path']
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    csv_logger = CSVLogger(os.path.join(save_path, "model_history_log.csv"), append=True)
    bi_model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds,
                 callbacks=[SaveLM(save_path), earlyStop_callback, csv_logger])
    save_loss_plot(save_path, "model_history_log.csv")
