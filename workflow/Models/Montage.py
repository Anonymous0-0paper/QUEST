import math
import random

import numpy as np

from workflow.DAG import DAG, DAGMode
from workflow.SubTask import SubTask


class Montage:

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

        level_1 = dag_size // 5
        level_2 = level_1 + level_1 // 2
        level_5 = dag_size - level_1 - level_2 - 6

        for i in range(level_1):
            predecessors = []
            for k in range(random.randint(1, 4)):
                predecessor = random.randint(level_1, level_1 + level_2 - 1)
                if predecessor not in predecessors:
                    predecessors.append(predecessor)

            for k in predecessors:
                edges.append([i, k, 0])

        for i in range(level_2):
            edges.append([level_1 + i, level_1 + level_2, 0])

        edges.append([level_1 + level_2, level_1 + level_2 + 1, 0])

        for i in range(level_5):
            edges.append([level_1 + level_2 + 1, level_1 + level_2 + 2 + i, 0])
            edges.append([level_1 + level_2 + 2 + i, level_1 + level_2 + 2 + level_5, 0])

        edges.append([level_1 + level_2 + 2 + level_5, level_1 + level_2 + 3 + level_5, 0])
        edges.append([level_1 + level_2 + 3 + level_5, level_1 + level_2 + 4 + level_5, 0])
        edges.append([level_1 + level_2 + 4 + level_5, level_1 + level_2 + 5 + level_5, 0])

        dag = DAG(DAGMode.Montage, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)
        dag.generate_communication_data_sizes(communication_min, communication_max)
        return dag
