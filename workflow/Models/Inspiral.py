import numpy as np

from workflow.DAG import DAGMode, DAG
from workflow.SubTask import SubTask


class Inspiral:
    @staticmethod
    def generate_dag(dag_id: int, dag_size: int,
                     memory_min: int, memory_max: int,
                     execution_min: int, execution_max: int,
                     deadline_min: int, deadline_max: int,
                     communication_min: int, communication_max: int) -> DAG:

        # generate tasks
        tasks: np.array(SubTask) = []
        for task_id in range(dag_size):
            task = SubTask.generate(dag_id, task_id, memory_min, memory_max, execution_min, execution_max)
            tasks.append(task)

        # generate communication
        edges: list[list[int]] = []

        # Calculate level sizes
        num_top_nodes = dag_size // 5  # Number of nodes in top level
        mid_layer_size = 5  # Fixed size for middle layer
        bottom_layer_size = num_top_nodes  # Same size as top layer

        # Connect top level to first convergence point
        convergence_point1 = num_top_nodes
        for i in range(num_top_nodes):
            edges.append([i, convergence_point1, 0])

        # Connect first convergence to middle layer
        mid_start = convergence_point1 + 1
        for i in range(mid_layer_size):
            edges.append([convergence_point1, mid_start + i, 0])

        # Connect middle layer to second convergence point
        convergence_point2 = mid_start + mid_layer_size
        for i in range(mid_layer_size):
            edges.append([mid_start + i, convergence_point2, 0])

        # Connect second convergence to bottom layer
        bottom_start = convergence_point2 + 1
        for i in range(bottom_layer_size):
            edges.append([convergence_point2, bottom_start + i, 0])

        # Connect bottom layer to final convergence point
        final_node = dag_size - 1
        for i in range(bottom_layer_size):
            edges.append([bottom_start + i, final_node, 0])

        dag = DAG(DAGMode.Inspiral, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)
        dag.generate_communication_data_sizes(communication_min, communication_max)
        return dag
