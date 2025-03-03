import numpy as np
import random

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class RL(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.assign = {}
        self.peak_aoi = self.dag.deadline / (len(self.dag.subtasks) * len(self.network.nodes) / 2)

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.05
        self.episodes = 20

        self.q_table = {}

        self.best_solution = None
        self.best_objective = float('inf')

    def run(self):
        for task_id in range(len(self.dag.subtasks)):
            for node_id in range(len(self.network.nodes)):
                self.q_table[(task_id, node_id)] = 0.0

        for episode in range(self.episodes):
            self.initialize_random()

            sorted_tasks = self.get_sorted_tasks()

            for task in sorted_tasks:
                if random.random() < self.epsilon:
                    node_id = random.randrange(len(self.network.nodes))
                else:
                    node_id = self.get_best_node(task.id)

                old_node_id = self.assign.get(task.id)
                self.assign[task.id] = node_id

                reward = self.calculate_reward()

                self.update_q_value(task.id, node_id, reward)

                if reward < 0 and old_node_id is not None:
                    self.assign[task.id] = old_node_id

            super().run()
            current_objective = self.calculate_combined_objective()

            if current_objective < self.best_objective:
                self.best_solution = self.assign.copy()
                self.best_objective = current_objective

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.best_solution:
            self.assign = self.best_solution

        super().run()

    def initialize_random(self):
        for i in range(len(self.dag.subtasks)):
            node_index = random.randrange(len(self.network.nodes))
            self.assign[i] = node_index

    def get_sorted_tasks(self):
        tasks = self.dag.subtasks.copy()
        tasks.sort(key=lambda x: x.execution[1])
        return tasks

    def get_best_node(self, task_id):
        best_node = 0
        best_q_value = float('-inf')

        for node_id in range(len(self.network.nodes)):
            q_value = self.q_table.get((task_id, node_id), 0.0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_node = node_id

        return best_node

    def calculate_reward(self):
        super().run()

        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        data_age = self.calculate_data_age()

        if data_age > self.peak_aoi:
            return -100.0

        reward = -(energy * 0.6 + completion_time * 0.4)

        return reward

    def update_q_value(self, task_id, node_id, reward):
        old_q = self.q_table.get((task_id, node_id), 0.0)

        new_q = old_q + self.alpha * (reward - old_q)

        self.q_table[(task_id, node_id)] = new_q

    def calculate_combined_objective(self):
        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        data_age = self.calculate_data_age()

        if data_age > self.peak_aoi:
            return float('inf')

        return energy * 0.6 + completion_time * 0.4