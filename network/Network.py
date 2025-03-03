import configparser
import math
import random

from network.Connection import Connection
from network.Node import Node, NodeType


class Network:
    def __init__(self, nodes: list[Node], connections: dict[tuple[str, str], Connection]):
        self.nodes = nodes
        self.connections = connections

    @staticmethod
    def generate():
        config = configparser.ConfigParser()
        config.read('config.ini')
        regions = [region.strip() for region in config['Regions']['region_names'].split(',')]
        connections = {}

        for region1 in regions:
            for region2 in regions:
                config_key_1 = f"{region1.lower()}_{region2.lower()}"
                config_key_2 = f"{region2.lower()}_{region1.lower()}"

                # Read connection parameters
                bandwidth = config['Connections'].get(f'{config_key_1}_bandwidth', None)
                min_latency = config['Connections'].get(f'{config_key_1}_min_latency', None)
                max_latency = config['Connections'].get(f'{config_key_1}_max_latency', None)

                if bandwidth is None:
                    bandwidth = config['Connections'][f'{config_key_2}_bandwidth']
                if min_latency is None:
                    min_latency = config['Connections'][f'{config_key_2}_min_latency']
                if max_latency is None:
                    max_latency = config['Connections'][f'{config_key_2}_max_latency']

                connections[(region1, region2)] = Connection(float(bandwidth), float(min_latency), float(max_latency))

        nodes = []
        node_id = 0

        for region in regions:
            region_lower = region.lower()
            node_types = [node_type.strip() for node_type in config['Nodes'][f'{region_lower}_types'].split(',')]

            for node_type in node_types:
                type_lower = node_type.lower()
                count_key = f'{region_lower}_{type_lower}_count'
                node_count = int(config['Nodes'][count_key])

                node_type_enum = getattr(NodeType, node_type)

                for i in range(node_count):
                    nodes.append(Node(
                        node_id,
                        node_type_enum,
                        config['Nodes'][f'{region_lower}_{type_lower}_server_model'],
                        region,
                        int(config['Nodes'][f'{region_lower}_{type_lower}_cpu_count']),
                        [float(speed.strip()) for speed in
                         config['Nodes'][f'{region_lower}_{type_lower}_cpu_speed'].split(',')],
                        int(config['Nodes'][f'{region_lower}_{type_lower}_ram']),
                        int(config['Nodes'][f'{region_lower}_{type_lower}_min_load']),
                        int(config['Nodes'][f'{region_lower}_{type_lower}_max_load'])
                    ))
                    node_id += 1

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
