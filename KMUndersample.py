import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


class KMUndersampling():
    def __init__(self, data, k=10):
        self.data = data
        self.k = k
        self.data_preprocessing()
        self.km_sampling()
    def data_preprocessing(self):
        '''Data pre-processing module, encoding
        classification variables'''
        self.discrete_variable = []
        self.discrete_map = {}
        self.discrete_map_r = {}
        for col in self.data.columns:
            class_code = []
            if self.data[col].dtype == object:
                self.discrete_variable.append(col)
                var = self.data[col].value_counts().index
                class_map = dict((v, code) for code, v in enumerate(var) if v is not '*')
                class_map_r = dict((code, v) for code, v in enumerate(var) if v is not '*')
                self.discrete_map[col] = class_map
                self.discrete_map_r[col] = class_map_r
                for v in self.data[col]:
                    code = self.discrete_map[col][v]
                    class_code.append(code)
                self.data[col] = class_code

    def calculate_distance(self, center_vector, sample_vectors):

            '''
            According to the clustering results of K-means, calculate the distance between the sample and the cluster center of the cluster, and return the sample index
            with the shortest distance from the cluster center
            :param center_vector: Center of cluster
            :param sample_vectors: Sample index of all vectors in the cluster
            :return: Sample index of the vector with the shortest distance from the center point
            '''
            distances = []
            samples = self.data.loc[sample_vectors]

            for row in range(len(samples)):
                distances.append(np.round(np.linalg.norm(center_vector - samples.iloc[row]), 2))
            min_distance_samp = sample_vectors[np.argmin(np.array(distances))]
            return min_distance_samp

    def km_sampling(self):
        '''
        Cluster under-sampling is performed according to the samples,
        and k samples are sampled from the samples as under-sampling results

        '''
        temp_samples = []
        self.KMeans = KMeans(n_clusters=self.k, random_state=0).fit(self.data)
        centers = self.KMeans.cluster_centers_
        for i_cluster in range(self.k):
            cluster_samples = np.where(self.KMeans.labels_ == i_cluster)[0].tolist()
            cluster_center = centers[i_cluster]
            cluster_sample = self.data.loc[self.calculate_distance(cluster_center, cluster_samples)]
            temp_samples.append(cluster_sample)
        self.undersamples = pd.DataFrame(columns=self.data.columns, data=temp_samples)
        for dis_col in self.discrete_variable:
            temp_class_type = []
            for v in self.undersamples[dis_col]:
                temp_class_type.append(self.discrete_map_r[dis_col][int(v)])
            self.undersamples.loc[:, dis_col] = temp_class_type
        self.sample_index = self.undersamples.index




