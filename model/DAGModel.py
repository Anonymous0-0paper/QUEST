from model.SubTaskModel import SubTaskModel


class DAGModel:
    def __init__(self, subtasks: list[SubTaskModel], edges: list[list[int]], deadline: int):
        self.subtasks = subtasks
        self.edges = edges  # [predecessor, successor, data size]
        self.deadline: int = deadline

    def clear(self):
        for subtask in self.subtasks:
            subtask.clear()

        for edge in self.edges:
            self.subtasks[edge[1]].total_data_needed += 1
