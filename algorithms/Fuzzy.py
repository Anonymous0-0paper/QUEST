import math

import numpy as np

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class Fuzzy(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.edge_threshold = 1 # GHz
        self.cloud_threshold = 2  # GHz

    def run(self):
        task_indices = list(range(len(self.dag.subtasks)))
        task_indices.sort(key=lambda x: self.dag.subtasks[x].execution_cost, reverse=True)

        for task_index in task_indices:
            intensity = self.calculate_task_intensity(task_index)

            # fuzzy membership values
            true_mem, false_mem = self.calculate_membership_values(task_index)

            if true_mem > false_mem / 2:
                node_index = self.select_suitable_node(intensity)
            else:
                node_index = self.select_suitable_node(math.inf)

            self.assign[task_index] = node_index

        super().run()

    def calculate_task_intensity(self, task_index):
        task = self.dag.subtasks[task_index]
        return task.execution_cost

    def calculate_membership_values(self, task_index):
        task = self.dag.subtasks[task_index]
        avg_comp = np.mean([t.execution_cost for t in self.dag.subtasks])
        max_comp = np.max([t.execution_cost for t in self.dag.subtasks])

        # Based on equations 21 and 22 from the paper
        true_membership = avg_comp / (avg_comp + max_comp + len(self.dag.subtasks))
        false_membership = avg_comp / (avg_comp + task.execution_cost + len(self.dag.subtasks))

        return true_membership, false_membership

    def select_suitable_node(self, intensity):
        available_nodes = []

        if intensity <= self.edge_threshold * 1_000_000_000:
            available_nodes = [i for i, node in enumerate(self.network.nodes)
                               if node.frequency < self.cloud_threshold]
        else:
            available_nodes = [i for i, node in enumerate(self.network.nodes)
                               if node.frequency >= self.cloud_threshold]

        if not available_nodes:
            available_nodes = list(range(len(self.network.nodes)))

        # Select node with minimum current load
        min_load: float = math.inf
        selected_node = available_nodes[0]

        for node_index in available_nodes:
            current_load = sum(1 for assignment in self.assign.values() if assignment == node_index)
            if current_load < min_load:
                min_load = current_load
                selected_node = node_index

        return selected_node
