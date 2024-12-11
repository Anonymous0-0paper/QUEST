import json
import sys

from openpyxl.reader.excel import load_workbook

from algorithms.Algorithm import Algorithm
from algorithms.Fuzzy import Fuzzy
from algorithms.NSGA3 import NSGA3
from algorithms.QUEST import QUEST
from algorithms.Random import Random
from model.DAGModel import DAGModel
from model.SubTaskModel import SubTaskModel
from network.Network import Network


class Experiment():

    def __init__(self, algorithms: list[str], loads: list[int], dag_files: dict[int, list[str]], iteration: int,
                 output: str):
        self.algorithms = algorithms
        self.loads = loads
        self.dag_files = dag_files
        self.iteration = iteration
        self.output = output

        # key: algorithm, load, [0: data age, 1: energy, 2: makespan, 3: load balance, 4: successrate]
        # value: samples
        self.result: dict[tuple[str, int, int], list[float]] = {}
        self.wb = load_workbook(self.output)

    def run(self):
        network = Network.generate()

        total = len(self.loads) * len(self.algorithms) * self.iteration
        for l in range(len(self.loads)):
            load = self.loads[l]

            for i in range(self.iteration):
                file_index = i % len(self.dag_files[load])
                dags: DAGModel | None = None
                with open(self.dag_files[load][file_index], 'r') as file:
                    json_data = json.load(file)
                    subtasks: list[SubTaskModel] = []
                    for subtask_dto in json_data["subtasks"]:
                        subtask = SubTaskModel(subtask_dto["id"], subtask_dto["executionCost"], subtask_dto["memory"])
                        subtasks.append(subtask)

                    dag = DAGModel(subtasks, json_data["edges"], json_data["deadline"])

                for a in range(len(self.algorithms)):
                    algorithm = self.algorithms[a]

                    current: float = (l * len(self.algorithms) * self.iteration + i * len(self.algorithms) + a)

                    alg: Algorithm | None = None
                    if algorithm == "Random":
                        alg = Random(network, dag)
                    elif algorithm == "Fuzzy":
                        alg = Fuzzy(network, dag)
                    elif algorithm == "NSGA3":
                        alg = NSGA3(network, dag)
                    elif algorithm == "QUEST":
                        alg = QUEST(network, dag)

                    alg.run()

                    if i == 0:
                        self.result[(algorithm, load, 0)] = []
                        self.result[(algorithm, load, 1)] = []
                        self.result[(algorithm, load, 2)] = []
                        self.result[(algorithm, load, 3)] = []
                        self.result[(algorithm, load, 4)] = []

                    self.result[(algorithm, load, 0)].append(alg.calculate_data_age())
                    self.result[(algorithm, load, 1)].append(alg.calculate_energy())
                    self.result[(algorithm, load, 2)].append(alg.calculate_completion_time())
                    self.result[(algorithm, load, 3)].append(alg.calculate_load_balance())
                    self.result[(algorithm, load, 4)].append(alg.calculate_success())

                    self.progress((current + 1) / total, algorithm)

    def store(self):

        ws = self.create_sheet("Data Age")
        row = 1
        for l in range(len(self.loads)):
            ws.cell(row=row, column=2 + l, value=self.loads[l])
        for algorithm in self.algorithms:
            row += 1
            ws.cell(row=row, column=1, value=algorithm)
            for l in range(len(self.loads)):
                data_age = sum(self.result[(algorithm, self.loads[l], 0)]) / self.iteration
                ws.cell(row=row, column=2 + l, value=data_age)

        ws = self.create_sheet("Energy")
        row = 1
        for l in range(len(self.loads)):
            for i in range(self.iteration):
                column = 2 + (l * self.iteration) + i
                ws.cell(row=row, column=column, value=self.loads[l])
        for algorithm in self.algorithms:
            row += 1
            ws.cell(row=row, column=1, value=algorithm)
            for l in range(len(self.loads)):
                for i in range(self.iteration):
                    column = 2 + (l * self.iteration) + i
                    energy = self.result[(algorithm, self.loads[l], 1)]
                    ws.cell(row=row, column=column, value=energy)

        ws = self.create_sheet("Makespan")
        row = 1
        for l in range(len(self.loads)):
            ws.cell(row=row, column=2 + l, value=self.loads[l])
        for algorithm in self.algorithms:
            row += 1
            ws.cell(row=row, column=1, value=algorithm)
            for l in range(len(self.loads)):
                makespan = sum(self.result[(algorithm, self.loads[l], 2)]) / self.iteration
                ws.cell(row=row, column=2 + l, value=makespan)

        ws = self.create_sheet("Load")
        row = 1
        for l in range(len(self.loads)):
            ws.cell(row=row, column=2 + l, value=self.loads[l])
        for algorithm in self.algorithms:
            row += 1
            ws.cell(row=row, column=1, value=algorithm)
            for l in range(len(self.loads)):
                load = sum(self.result[(algorithm, self.loads[l], 3)]) / self.iteration
                ws.cell(row=row, column=2 + l, value=load)

        ws = self.create_sheet("Success Rate")
        row = 1
        for l in range(len(self.loads)):
            ws.cell(row=row, column=2 + l, value=self.loads[l])
        for algorithm in self.algorithms:
            row += 1
            ws.cell(row=row, column=1, value=algorithm)
            for l in range(len(self.loads)):
                successrate = sum(self.result[(algorithm, self.loads[l], 4)]) / self.iteration
                ws.cell(row=row, column=2 + l, value=successrate)

        self.wb.save(self.output)

    def create_sheet(self, sheet_name: str):
        if sheet_name in self.wb.sheetnames:
            ws = self.wb[sheet_name]
        else:
            ws = self.wb.create_sheet(sheet_name)

        return ws

    def progress(self, percent: float, algorithm: str):
        arrow = '=' * int(round(percent * 100) - 1)
        spaces = ' ' * (100 - len(arrow))
        sys.stdout.write(f'\rProgress: [{arrow + spaces}] {int(percent * 100)}% [{algorithm}]')
        sys.stdout.flush()
