# Mahdi Abdollahpour, 08/04/2022, 12:42 AM, PyCharm, lgeconvrec


import numpy as np
from tensorflow import keras
import os


class SaveLM(keras.callbacks.Callback):

    def __init__(self, save_path="local-tf-checkpoint"):
        super(SaveLM, self).__init__()
        self.best_weights = None
        self.save_path = save_path
        sp = os.path.join(self.save_path, 'LM')
        self.sp = sp

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.best_epoch = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

            self.model.passage_encoder.passage_encoder.save_pretrained(self.sp)

    # def on_train_end(self, logs=None):
    #     if self.stopped_epoch > 0:
    #         print("Best checkpoint at" % (self.best_epoch + 1))

class SaveLMAndItemEmbedding(keras.callbacks.Callback):

    def __init__(self, save_path="local-tf-checkpoint"):
        super(SaveLMAndItemEmbedding, self).__init__()
        self.best_weights = None
        self.save_path = save_path
        sp = os.path.join(self.save_path, 'LM')
        self.sp = sp

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.best_epoch = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

            self.model.passage_encoder.passage_encoder.save_pretrained(self.sp)
            with open(os.path.join(self.save_path, 'id_to_index.json'), 'wb') as f:
                f.write(str.encode(str(self.model.id_to_index)))
            embs = self.model.get_embeddings()
            with open(os.path.join(self.save_path, 'item_embeddings.npy'), 'wb') as f:
                np.save(f, embs)

    # def on_train_end(self, logs=None):
    #     if self.stopped_epoch > 0:
    #         print("Best checkpoint at" % (self.best_epoch + 1))
