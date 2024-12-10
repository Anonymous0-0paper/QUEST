class SubTaskModel:
    def __init__(self, subtask_id: int, execution_cost: int, memory: int):
        self.id = subtask_id
        self.execution_cost = execution_cost  # MI
        self.memory = memory

        self.data_received_time: int = 0
        self.total_data_needed: int = 0

        # 0: node id, 1: start time, 2: end time
        self.execution = list[int]

    def clear(self):
        self.execution = []
        self.data_received_time = 0
        self.total_data_needed = 0
