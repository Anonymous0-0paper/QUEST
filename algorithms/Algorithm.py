import math
from queue import Queue

from model.DAGModel import DAGModel
from model.SubTaskModel import SubTaskModel
from network.Network import Network
from network.Node import Node, NodeType


class Algorithm:

    def __init__(self, network: Network, dag: DAGModel):
        self.network = network
        self.dag = dag
        self.assign: dict[int, int] = {}
        self.frequencies: dict[int, float] = {}

        self.total_latency = 0

    def clear(self):
        self.network.clear()
        self.dag.clear()
        self.total_latency = 0

    def run(self):
        self.clear()

        queue = Queue()
        backup_queue = Queue()

        for subtask in self.dag.subtasks:
            queue.put(subtask)

        while not queue.empty():
            qsize = queue.qsize()
            for _ in range(qsize):
                subtask: SubTaskModel = queue.get()
                if subtask.total_data_needed == 0:
                    node = self.network.nodes[self.assign[subtask.id]]
                    frequency = self.frequencies.get(subtask.id, None)
                    self.schedule(subtask, node, frequency)
                    self.send_data(subtask)
                else:
                    backup_queue.put(subtask)
            while not backup_queue.empty():
                queue.put(backup_queue.get())

    @staticmethod
    def schedule(subtask: SubTaskModel, node: Node, frequency: float = None):

        candidate_core_index = 0
        candidate_core_idle_time = math.inf
        for c in range(len(node.allocations)):
            if len(node.allocations[c]) == 0:
                candidate_core_index = c
                candidate_core_idle_time = 0
                break
            core = node.allocations[c]
            if candidate_core_idle_time > core[-1][2]:
                candidate_core_index = c
                candidate_core_idle_time = core[-1][2]

        core = node.allocations[candidate_core_index]
        start_time = max(subtask.data_received_time, candidate_core_idle_time)
        finish_time = start_time + node.get_execution_time(subtask.execution_cost, frequency)
        core.append([subtask.id, start_time, finish_time])
        subtask.execution = [node.index, start_time, finish_time]

    def send_data(self, subtask: SubTaskModel):
        edges = [e for e in self.dag.edges if e[0] == subtask.id]
        for e in edges:
            data_transfer_time = self.network.transition(self.assign[e[0]], self.assign[e[1]], e[2])
            self.total_latency += data_transfer_time
            data_received_time = subtask.execution[2] + data_transfer_time
            self.dag.subtasks[e[1]].total_data_needed -= 1
            if self.dag.subtasks[e[1]].data_received_time < data_received_time:
                self.dag.subtasks[e[1]].data_received_time = data_received_time

    def calculate_completion_time(self):
        makespan = 0
        for subtask in self.dag.subtasks:
            if makespan < subtask.execution[2]:
                makespan = subtask.execution[2]

        return makespan

    def calculate_energy(self):
        energy_dynamic = 0.0
        energy_static = 0.0
        makespan = self.calculate_completion_time()
        for node in self.network.nodes:
            energy_static += node.static_power * makespan
            power_dif = (node.dynamic_power - node.static_power) / node.cores
            for core in node.allocations:
                for item in core:
                    frequency = self.frequencies.get(item[0], None)
                    if frequency is None:
                        energy_dynamic += power_dif * (item[2] - item[1])
                    else:
                        max_frequency = node.frequencies[0]
                        rate = frequency / max_frequency
                        time = item[2] - item[1]
                        energy_dynamic += power_dif * time * (rate ** 3)
                        energy_static -= (node.static_power * time) * (1 - rate)

        return energy_dynamic + energy_static

    def calculate_data_age(self):
        age: float = 0.0
        for subtask in self.dag.subtasks:
            age += subtask.execution[2] - subtask.data_received_time

        return age / len(self.dag.subtasks)

    def calculate_latency(self):
        return self.total_latency / len(self.dag.subtasks)

    def calculate_load(self):
        edge_load: float = 0.0
        edge_count: int = 0
        cloud_load: float = 0.0
        cloud_count: int = 0

        for node in self.network.nodes:
            node_load: float = 0.0
            for core in node.allocations:
                for item in core:
                    node_load += item[2] - item[1]

            if node.node_type == NodeType.Edge:
                edge_load += node_load
                edge_count += 1
            else:
                cloud_load += node_load
                cloud_count += 1

        return edge_load / edge_count, cloud_load / cloud_count

    def calculate_load_per_node(self):
        load: list[float] = []

        for node in self.network.nodes:
            node_load: float = 0.0
            for core in node.allocations:
                for item in core:
                    node_load += item[2] - item[1]

            load.append(node_load)

        return load

    def calculate_success(self):
        makespan = self.calculate_completion_time()
        return makespan <= self.dag.deadline
