import numpy as np
import csv
import networkx as nx
import random
from gensim.models import Word2Vec
import pandas as pd


class Embedding:
    """
    Implements an embedding of the graph based on Random Walks.
    """
    def __init__(self, time_slot=4, num_walks=1000,
                 walk_length=100, embedding_size=20):
        self.time_slot = time_slot
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.num_zone = 180
        self.Walks = []

    def spatial_graph(self, path='data/drive_dist.npy'):
        Distance = np.load(path, allow_pickle=True)
        self.num_zone = len(Distance)
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
        data = pd.read_csv('data/taxi_flow_2016-01.csv',
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

    def random_walk(self, Graph, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(Graph, walk_length, v))
        return walks

    def deepwalk_walk(self, Graph, walk_length, start_node):
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
        nodes = np.arange(self.num_zone*int(24/self.time_slot))
        print("Nodes {}".format(len(nodes)))
        print("Walking through the spatial graph...")
        Walks_SG = self.random_walk(self.spatial_embedding, nodes, self.num_walks, self.walk_length)
        print("Walking through the flow graph...")
        Walks_FG = self.random_walk(self.taxi_embedding, nodes, self.num_walks, self.walk_length)
        print("Combine the spatial and flow  graph...")
        Walks = Walks_SG+Walks_FG
        self.Walks = np.array(Walks)
        return self.Walks

    def word_to_vec(self):
        Str_Walks = []
        Walks_shape = np.shape(self.Walks)
        for line in self.Walks:
            Walk = []
            for item in line:
                Walk.append(str(item))
            Str_Walks.append(Walk)

        w2v_model = Word2Vec(Str_Walks, size=self.embedding_size)
    #    w2v_model.save('../data/model')
    #
        model = w2v_model
    #    model = Word2Vec.load('../data/model')
        ZoneEmbed = []
        for N in nodes:
            ZoneEmbed.append(model[str(N)])
        ZoneEmbed = np.array(ZoneEmbed)
        return ZoneEmbed

if __name__ == '__main__':
    print("Create embedding instance")
    zone_embedding = Embedding()
    print("Create spatial graph")
    zone_embedding.spatial_graph()
    print("Create taxi_flow graph")
    zone_embedding.taxi_flow()
    print("Create random walks")
    zone_embedding.simulate_random_walks()
#     emb = zone_embedding.word_to_vec()
#     print(emb)
