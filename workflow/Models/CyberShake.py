import numpy as np

from workflow.DAG import DAG, DAGMode
from workflow.SubTask import SubTask


class CyberShake:
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

        # Calculate levels
        entry_nodes = 2  # Number of nodes at the top level
        second_level_per_entry = 4  # Number of children per entry node
        third_level_size = entry_nodes * second_level_per_entry  # Total nodes in third level
        fourth_level_size = third_level_size  # Same size as third level

        # Connect entry nodes to second level
        for i in range(entry_nodes):
            start_idx = i * second_level_per_entry + entry_nodes
            for j in range(second_level_per_entry):
                edges.append([i, start_idx + j, 0])

        # Connect second level to third level
        second_level_start = entry_nodes
        third_level_start = second_level_start + third_level_size

        # Cross connections between second and third level
        for i in range(second_level_per_entry * entry_nodes):  # Total size of second level
            for j in range(third_level_size):
                edges.append([second_level_start + i, third_level_start + j, 0])

        # Connect third level to exit node
        exit_node = dag_size - 1
        for i in range(third_level_size):
            edges.append([third_level_start + i, exit_node, 0])

        dag = DAG(DAGMode.CyberShake, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)
        dag.generate_communication_data_sizes(communication_min, communication_max)
        return dag
