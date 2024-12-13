import random

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class Greedy(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.peak_aoi = self.dag.deadline / len(self.dag.subtasks) * (len(self.network.nodes) / 2)
        self.assign = {}

    def run(self):
        self.initialize()
        super().run()

        sorted_tasks = self.get_sorted_tasks()

        for task in sorted_tasks:
            best_node = None
            best_objective = float('inf')

            for node in self.network.nodes:
                self.assign[task.id] = node.index

                objectives = self.calculate_objectives()
                energy_cost = objectives["energy"]
                data_age = objectives["data_age"]
                objective = energy_cost / 1000 + data_age * 1000

                if data_age <= self.peak_aoi:
                    if objective < best_objective:
                        best_objective = objective
                        best_node = node.index

            if best_node is not None:
                self.assign[task.id] = best_node

        super().run()

    def initialize(self):
        for i in range(len(self.dag.subtasks)):
            node_index = random.randrange(len(self.network.nodes))
            self.assign[i] = node_index

    def get_sorted_tasks(self):
        tasks = self.dag.subtasks.copy()
        tasks.sort(key=lambda x: x.execution[1])
        return tasks

    def calculate_objectives(self):
        super().run()
        energy = self.calculate_energy()
        data_age = self.calculate_data_age()

        return {
            'energy': energy,
            'data_age': data_age
        }
