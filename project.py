import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans


class taxi:
    def __init__(self, time_slot=4, num_walks=1000,
                 walk_length=100, embedding_size=20):
        '''
        Init method, input parameters for the analysis
        '''
        self.time_slot = time_slot
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.num_zone = 180
        self.nodes = np.arange(self.num_zone*int(24/self.time_slot))

    def spatial_graph(self, path='data/drive_dist.npy'):
        '''
        Create spatial graph from average driving distance data
        '''
        Distance = np.load(path, allow_pickle=True)
        SpatialGraph = np.exp(-Distance)

        ExpandedSpatial = np.zeros((int(24/self.time_slot)*self.num_zone,
                                    int(24/self.time_slot)*self.num_zone))
        for index_1 in range(int(24/self.time_slot)):
            for index_2 in range(int(24/self.time_slot)):
                ExpandedSpatial[index_1*self.num_zone:(index_1+1)*self.num_zone,
                                index_2*self.num_zone:(index_2+1)*self.num_zone] = SpatialGraph

        self.spatial_embedding = nx.DiGraph(ExpandedSpatial)
        return self.spatial_embedding

    def taxi_flow(self, path='data/taxi_flow_2016-01.csv'):
        '''
        Create taxi embedding from taxi data
        '''
        data = pd.read_csv(path,
                           parse_dates=[0, 1],
                           usecols=['stime', 'ttime', 'sregion', 'tregion'])

        data['shour'] = data['stime'].dt.hour
        data['thour'] = data['ttime'].dt.hour
        FlowGraph = np.zeros((int(24/self.time_slot)*self.num_zone,
                              int(24/self.time_slot)*self.num_zone))
        for _, row in data.iterrows():
            FlowGraph[
                row['sregion']+int(row['shour']/self.time_slot)*self.num_zone,
                row['tregion']+int(row['thour']/self.time_slot)*self.num_zone] \
                = FlowGraph[
                row['sregion']+int(row['shour']/self.time_slot)*self.num_zone,
                row['tregion']+int(row['thour']/self.time_slot)*self.num_zone] + 1
        self.taxi_embedding = nx.DiGraph(FlowGraph)
        return self.taxi_embedding

    @staticmethod
    def random_walk(Graph, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(taxi.deepwalk_walk(Graph, walk_length, v))
        return walks

    @staticmethod
    def deepwalk_walk(Graph, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(Graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_random_walks(self):
        '''
        Radom walk on Taxi embedding and Spatial embedding
        '''
        Walks_SG = self.random_walk(self.spatial_embedding, self.nodes,
                                    self.num_walks, self.walk_length)
        Walks_FG = self.random_walk(self.taxi_embedding, self.nodes,
                                    self.num_walks, self.walk_length)
        Walks = Walks_SG+Walks_FG
        self.Walks = np.array(Walks)
        return self.Walks

    def wordtoVec(self):
        '''
        Word to Vec algorithm to extract features for each nodes at each time
        frame from the random walks
        '''
        Str_Walks = []
        for line in self.Walks:
            Walk = []
            for item in line:
                Walk.append(str(item))
            Str_Walks.append(Walk)

        model = Word2Vec(Str_Walks, size=self.embedding_size)
        ZoneEmbed = []
        for N in self.nodes:
            ZoneEmbed.append(model[str(N)])
        ZoneEmbed = np.array(ZoneEmbed)
        ZoneEmbed_shape = np.shape(ZoneEmbed)

        emb = np.zeros((self.num_zone,
                        int(ZoneEmbed_shape[0]/self.num_zone*ZoneEmbed_shape[1])))
        for index_1 in range(ZoneEmbed_shape[0]):
            for index_2 in range(ZoneEmbed_shape[1]):
                emb[index_1 % self.num_zone, index_2 + ZoneEmbed_shape[1]*int(index_1/self.num_zone)] \
                    = ZoneEmbed[index_1, index_2]
        self.ZoneEmbed = emb
        return self.ZoneEmbed

    @staticmethod
    def regression(X_train, y_train, X_test, alpha):
        """
        Specify and fit a Ridge Linear Regression model
        Returns the predicted value of Y
        """
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        return y_pred

    @staticmethod
    def kf_predict(X, Y, alpha):
        """
        Run regressions with cross validations
        Returns results from various runs of K-Fold cross validations
        """
        kf = KFold(n_splits=5)
        y_preds = []
        y_truths = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            y_pred = taxi.regression(X_train, y_train, X_test, alpha)
            y_preds.append(y_pred)
            y_truths.append(y_test)
        return np.concatenate(y_preds), np.concatenate(y_truths)

    @staticmethod
    def compute_metrics(y_pred, y_test):
        """
        Compute prediction accuracy metrics
        """
        y_pred[y_pred < 0] = 0

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        msle = mean_squared_error(np.log(y_test+1), np.log(y_pred+1))
        return mae, rmse, r2, msle

    @staticmethod
    def predict_crime(emb, target):
        """
        Run a prediction
        """
        maes, rmses, r2s, msles = [], [], [], []
        for a in [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 2, 5, 10, 20]:
            y_pred, y_test = taxi.kf_predict(emb, target, a)
            mae, rmse, r2, msle = taxi.compute_metrics(y_pred, y_test)
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
            msles.append(msle)
        print("MAE: ", np.min(maes))
        print("RMSE: ", np.min(rmses))
        print("R2: ", np.max(r2s))
        print("MSLE: ", np.min(msles))
        return np.min(maes), np.min(rmses), np.max(r2s), np.min(msles)

    def run_crime_prediction(self):
        """
        Predict number of crime incedences
        Make use of embeddings matrix to predict crime
        """
        crime = np.load("data/crime_counts-2015.npy")
        crime = crime.reshape(-1)

        mae, rmse, r2, msle = taxi.predict_crime(self.ZoneEmbed, crime)

        res = {"mae": mae, "rmse": rmse, "r2": r2, "msle": msle}
        return res

    def kmeans(self, K=12):
        '''
        K-means on the ZoneEmbed data
        '''
        kmeans = KMeans(n_clusters=K, random_state=3)
        emb_labels = kmeans.fit_predict(self.ZoneEmbed)
        return emb_labels
