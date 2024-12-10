import math
import random

from network.Node import Node, NodeType
from network.Connection import Connection


class Network:
    def __init__(self, nodes: list[Node], connections: dict[tuple[str, str], Connection]):
        self.nodes = nodes
        self.connections = connections

    @staticmethod
    def generate():
        connections = {
            ("Lyon", "Lyon"): Connection(1, 6.88, 25.8),
            ("Luxembourg", "Luxembourg"): Connection(10, 25.9, 34.5),
            ("Toulouse", "Toulouse"): Connection(10, 9.63, 34.4),

            ("Lyon", "Luxembourg"): Connection(10, 9.63, 34.4),
            ("Lyon", "Toulouse"): Connection(10, 9.63, 34.4),
            ("Luxembourg", "Toulouse"): Connection(10, 9.63, 34.4),
        }
        keys = [(key[0], key[1]) for key in connections.keys()]
        for key in keys:
            if key[0] != key[1]:
                connections[(key[1], key[0])] = connections[key]

        nodes = []
        nodes.extend([Node(i, NodeType.Edge, "Sun Fire V20z", "Lyon",
                           1, 2.4, 2, 20, 40) for i in range(10)])
        nodes.extend([Node(i, NodeType.Edge, " Dell PowerEdge M620", "Luxembourg",
                           6, 2, 32, 20, 40) for i in range(10, 20)])
        nodes.extend([Node(i, NodeType.Cloud, "HPE Proliant DL360 Gen10+", "Toulouse",
                           16, 2.4, 256, 40, 80) for i in range(20, 22)])

        network = Network(nodes, connections)
        return network

    def clear(self):
        for node in self.nodes:
            node.clear()

    # ms
    def transition(self, source_id: int, destination_id: int, data: float):
        source = self.nodes[source_id]
        destination = self.nodes[destination_id]
        connections = []
        if source.region == destination.region:
            connections = [self.connections[source.region, source.region]]
        else:
            connections = [
                self.connections[source.region, source.region],
                self.connections[source.region, destination.region],
                self.connections[destination.region, destination.region],
            ]

        transition_time = 0
        for connection in connections:
            latency = random.random() * (connection.max_latency - connection.min_latency) + connection.min_latency
            transition_time += data * 8 / connection.bandwidth + latency

        return math.ceil(transition_time)