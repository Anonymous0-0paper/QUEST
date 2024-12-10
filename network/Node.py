import math


class NodeType[Enum]:
    Cloud = "Cloud",
    Edge = "Edge",


class Node:
    def __init__(self, index: int, node_type: NodeType, name: str, region: str, cores: int,
                 frequency: float, ram: float, static_power: float, dynamic_power: float):
        self.index = index
        self.node_type = node_type
        self.name = name
        self.region = region
        self.cores = cores
        self.frequency = frequency  # in GHz
        self.ram = ram  # in GiB
        self.static_power = static_power
        self.dynamic_power = dynamic_power

        # 0: task id, 1: start time, 2: end time
        self.allocations: list[list[list[int]]] = [[] for _ in range(cores)]

    def clear(self):
        self.allocations = [[] for _ in range(self.cores)]

    def get_execution_time(self, execution_cost: int) -> int:
        return math.ceil(execution_cost / self.frequency)  # ms
