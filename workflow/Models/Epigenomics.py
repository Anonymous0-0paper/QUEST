import numpy as np

from workflow.DAG import DAGMode, DAG
from workflow.SubTask import SubTask


class Epigenomics:
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
        num_branches = 4  # Number of parallel branches
        nodes_per_branch = (dag_size - 2) // num_branches  # Nodes per branch excluding entry and exit

        # Entry node (node 0) connections
        for i in range(num_branches):
            start_idx = 1 + i * nodes_per_branch
            edges.append([0, start_idx, 0])  # Connect entry to first node of each branch

            # Create the branch
            for j in range(nodes_per_branch - 1):
                current_idx = start_idx + j
                next_idx = current_idx + 1
                edges.append([current_idx, next_idx, 0])

            # Connect last node of branch to exit node
            edges.append([start_idx + nodes_per_branch - 1, dag_size - 1, 0])

        dag = DAG(DAGMode.Epigenomics, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)
        dag.generate_communication_data_sizes(communication_min, communication_max)
        return dag
