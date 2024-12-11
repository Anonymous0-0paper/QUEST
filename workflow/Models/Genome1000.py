import numpy as np

from workflow.DAG import DAG, DAGMode
from workflow.SubTask import SubTask


class Genome1000:

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

        level_1 = (dag_size - 2) // 3
        level_3 = (dag_size - 2) - level_1

        for i in range(level_1):
            edges.append([i, level_1, 0])

        for i in range(level_3):
            edges.append([level_1, level_1 + 2 + i, 0])
            edges.append([level_1 + 1, level_1 + 2 + i, 0])

        dag = DAG(DAGMode.Genome1000, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)
        dag.generate_communication_data_sizes(communication_min, communication_max)
        return dag
