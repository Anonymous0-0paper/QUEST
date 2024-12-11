import random

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class Random(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)

    def run(self):
        for i in range(len(self.dag.subtasks)):
            node_index = random.randint(0, len(self.network.nodes) - 1)
            self.assign[i] = node_index

        super().run()
