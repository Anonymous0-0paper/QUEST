# import random
# from algorithms.Algorithm import Algorithm
# from model.DAGModel import DAGModel
# from network.Network import Network
#
#
# class GreedyEnergyMinimization(Algorithm):
#     def __init__(self, network: Network, dag: DAGModel, peak_aoi: int):
#         """
#         Initialize the Greedy Energy Minimization algorithm with the network, DAG model, and Peak AoI constraint.
#         """
#         super().__init__(network, dag)
#         self.peak_aoi = peak_aoi  # Age of Information (AoI) constraint in milliseconds
#         self.assign = None  # Task-to-node assignment map
#         self.energy_costs = []  # Store energy costs for debugging/reporting
#         self.completion_times = []  # Store task completion times for debugging/reporting
#         self.debug_mode = True  # Enable or disable debug logging
#
#     def run(self):
#         """
#         Execute the greedy energy minimization algorithm. This method initializes
#         the greedy assignment and performs all necessary steps to compute task assignment.
#         """
#         if self.debug_mode:
#             print("Starting Greedy Energy Minimization Algorithm...")
#
#         # Perform the greedy assignment logic
#         self.assign = self.greedy_assignment()
#
#         if self.debug_mode:
#             print("Final assignment:", self.assign)
#
#         # Execute the base class `run` to finalize the scheduling
#         super().run()
#
#     def greedy_assignment(self):
#         """
#         Greedy assignment of tasks to nodes to minimize energy consumption
#         while respecting the Peak AoI constraint. This method iteratively assigns tasks
#         to nodes based on energy minimization and checks feasibility.
#         """
#         num_tasks = len(self.dag.subtasks)  # Total number of tasks in the DAG
#         num_nodes = len(self.network.nodes)  # Total number of available nodes in the network
#         assignment = {}  # Dictionary to store task-to-node assignment
#
#         if self.debug_mode:
#             print(f"Number of tasks: {num_tasks}, Number of nodes: {num_nodes}")
#
#         # Iterate over each task in the DAG
#         for task_id in range(num_tasks):
#             best_node = None  # Track the best node for the current task
#             best_energy = float('inf')  # Initialize the best energy cost to infinity
#
#             if self.debug_mode:
#                 print(f"\nEvaluating task {task_id}...")
#
#             # Iterate over each node in the network
#             for node_id in range(num_nodes):
#                 # Create a temporary assignment map for evaluation
#                 temp_assignment = assignment.copy()
#                 temp_assignment[task_id] = node_id  # Assign current task to the current node
#                 self.assign = temp_assignment  # Update self.assign temporarily
#
#                 # Calculate objectives for the current temporary assignment
#                 objectives = self.calculate_objectives()
#                 energy_cost = objectives["energy"]
#                 completion_time = objectives["completion_time"]
#
#                 if self.debug_mode:
#                     print(f"  Node {node_id}: Energy = {energy_cost}, Completion Time = {completion_time}")
#
#                 # Check if the assignment satisfies the AoI constraint and minimizes energy
#                 if completion_time <= self.peak_aoi:
#                     if energy_cost < best_energy:
#                         best_energy = energy_cost
#                         best_node = node_id
#
#                         if self.debug_mode:
#                             print(f"    -> New best node for task {task_id}: Node {best_node} with energy {best_energy}")
#
#             # Assign the best node for the current task
#             if best_node is not None:
#                 assignment[task_id] = best_node
#                 if self.debug_mode:
#                     print(f"Task {task_id} assigned to Node {best_node} with energy cost {best_energy}")
#             else:
#                 # If no valid assignment is found, mark the task as unassigned
#                 assignment[task_id] = -1  # Use -1 to indicate no valid assignment
#                 if self.debug_mode:
#                     print(f"Task {task_id} could not be assigned to any node (constraint violation).")
#
#         if self.debug_mode:
#             print("\nFinal Assignment:", assignment)
#
#         return assignment
#
#     def calculate_objectives(self):
#         """
#         Compute the objectives for energy consumption and completion time based on the current task assignment.
#         This method simulates the calculation of these metrics with added redundancies for demonstration.
#         """
#         energy = self.calculate_energy()
#         completion_time = self.calculate_completion_time()
#
#         if self.debug_mode:
#             print(f"Calculated Objectives: Energy = {energy}, Completion Time = {completion_time}")
#
#         return {
#             'energy': energy / 1000,
#             'completion_time': completion_time
#         }
#
#     def calculate_energy(self):
#         """
#         Simulate the energy calculation based on the current assignment.
#         This is a placeholder method that adds complexity for demonstration purposes.
#         """
#         total_energy = 0
#         for task_id, node_id in (self.assign or {}).items():
#             if node_id != -1:  # Check if the task is assigned
#
#                 task_energy = random.uniform(10, 50) * (task_id + 1) / (node_id + 1)
#                 total_energy += task_energy
#                 if self.debug_mode:
#                     print(f"  Task {task_id} on Node {node_id}: Energy = {task_energy}")
#             else:
#                 if self.debug_mode:
#                     print(f"  Task {task_id} is unassigned. Skipping energy calculation.")
#
#         if self.debug_mode:
#             print(f"Total Energy Cost: {total_energy}")
#
#         return total_energy
#
#     def calculate_completion_time(self):
#         """
#         Simulate the completion time calculation based on the current assignment.
#         This is a placeholder method that adds complexity for demonstration purposes.
#         """
#         max_completion_time = 0
#         for task_id, node_id in (self.assign or {}).items():
#             if node_id != -1:  # Check if the task is assigned
#
#                 task_time = random.uniform(5, 20) * (task_id + 1) / (node_id + 1)
#                 max_completion_time = max(max_completion_time, task_time)
#                 if self.debug_mode:
#                     print(f"  Task {task_id} on Node {node_id}: Completion Time = {task_time}")
#             else:
#                 if self.debug_mode:
#                     print(f"  Task {task_id} is unassigned. Skipping completion time calculation.")
#
#         if self.debug_mode:
#             print(f"Maximum Completion Time: {max_completion_time}")
#
#         return max_completion_time

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