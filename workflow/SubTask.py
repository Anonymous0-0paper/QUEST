import random
import json


class SubTask:

    def __init__(self, dag_id: int, subtask_id: int):
        self.dag_id = dag_id
        self.id: int = subtask_id
        self.execution_cost: int | None = None  # MI
        self.memory: int | None = None

    @staticmethod
    def generate(dag_id: int, subtask_id: int, memory_min: int, memory_max: int,
                 execution_min: int, execution_max: int):
        task = SubTask(dag_id, subtask_id)
        task.generate_memory(memory_min, memory_max)
        task.generate_execution_cost(execution_min, execution_max)
        return task

    def generate_memory(self, memory_min: int, memory_max: int):
        self.memory = random.randint(memory_min, memory_max)

    def generate_execution_cost(self, execution_min: int, execution_max: int):
        self.execution_cost = random.randint(execution_min, execution_max)

class SubTaskEncoder(json.JSONEncoder):

    def default(self, obj: SubTask):
        if isinstance(obj, SubTask):
            return {
                'id': obj.id,
                'executionCost': obj.execution_cost,
                'memory': obj.memory,
            }
        return super().default(obj)
