from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import pickle
import math
from Neural_PM.prefernce_matching.LM import *
from Neural_PM.clustering.preprocess import preprocess_text_df


class VectorizerModel():
    def fit_transform(self, texts):
        pass

    def apply_batch(self, X, strategy=None):
        def save(self, path):
            pass


class TFIDFVectorizerModel(VectorizerModel):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts, strategy=None):
        texts = [preprocess_text_df(x) for x in texts]
        X = self.vectorizer.fit_transform(texts)
        return X

    def apply_batch(self, X, strategy=None):
        return self.vectorizer.transform(X).toarray()

    def save(self, path):
        pickle.dump(self.vectorizer, open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb'))  # Saving the model

    def __str__(self):
        return 'TFIDF'


class BERTVectorizerModel(VectorizerModel):
    def __init__(self, bert_name, strategy):
        self.bert_name = bert_name
        self.embedder = BERT_model(self.bert_name, self.bert_name, True)
        self.strategy = strategy
        self.CHUNK_SIZE = 800

    def __str__(self):
        return self.bert_name.replace('./', '#').replace('/', '#').replace('-', '#').replace('_', '$')

    def get_features(self, texts):

        if self.strategy is not None and len(texts) > self.CHUNK_SIZE:
            embeds = []
            chunks = math.ceil(len(texts) / self.CHUNK_SIZE)
            for i in range(chunks):
                if i != chunks - 1:
                    rev_e = self.embedder.apply_batch(texts[i * self.CHUNK_SIZE:(i + 1) * self.CHUNK_SIZE],
                                                      self.strategy)
                else:
                    rev_e = self.embedder.apply_batch(texts[i * self.CHUNK_SIZE:], self.strategy)
                # print(rev_e.reshape(-1, 768).shape)
                rev_e = np.array(rev_e)
                embeds.append(rev_e.reshape(-1, 768))

            rev_embed = np.concatenate(embeds, axis=0)
        else:
            rev_embed = self.embedder.apply_batch(texts, self.strategy)
        return rev_embed

    def save(self, path):
        pass



