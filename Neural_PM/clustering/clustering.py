from sklearn.cluster import KMeans
import os
import pickle


class ClusterReviews():

    def __init__(self, n_clusters=50,):

        self.n_clusters = n_clusters
        # self.vectorizermodel = vectorizermodel
        self.path = 'clustering_' + str(self.n_clusters)
                    # + '_' + vectorizermodel.__str__()
        # self.new_data_path = os.path.join(self.path, 'data_with_cluster_labels.csv')

    def cluster(self,X):

        print('Kmeans...')
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=100).fit(X)
        self.labels = self.kmeans.predict(X)
        return self.labels


    def save(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        pickle.dump(self.kmeans, open(os.path.join(self.path, 'kmeans.pkl'), 'wb'))  # Saving the model
        # pickle.dump(X, open(os.path.join(self.path, 'X.pkl'), 'wb'))  # Saving the model

        # self.vectorizermodel.save(self.path)

    def load(self):
        pass
