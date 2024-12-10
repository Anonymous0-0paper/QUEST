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

mode = "montage"
loads = [50, 100, 150]
algorithms = ["Random", "Fuzzy", "NSGA3", "QUEST"]
dag_files = {
    50: [f"./workflow/Outputs/montage-50/dag-{i + 1}.json" for i in range(100)],
    100: [f"./workflow/Outputs/montage-100/dag-{i + 1}.json" for i in range(100)],
    150: [f"./workflow/Outputs/montage-150/dag-{i + 1}.json" for i in range(100)],
}
iteration = 10
output = f"./result-{mode}.xlsx"

wb = load_workbook(output)


def create_sheet(sheet_name: str):
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)

    return ws


def progress(percent: float):
    arrow = '=' * int(round(percent * 100) - 1)
    spaces = ' ' * (100 - len(arrow))
    sys.stdout.write(f'\rProgress: [{arrow + spaces}] {int(percent * 100)}% [{algorithm}]')
    sys.stdout.flush()


if __name__ == '__main__':
    # key: algorithm, load, [0: data age, 1: energy, 2: makespan, 3: load balance, 4: successrate]
    # value: samples
    result: dict[tuple[str, int, int], list[float]] = {}
    network = Network.generate()

    total = len(loads) * len(algorithms) * iteration
    for l in range(len(loads)):
        load = loads[l]

        for i in range(iteration):
            file_index = i % len(dag_files[load])
            dags: DAGModel | None = None
            with open(dag_files[load][file_index], 'r') as file:
                json_data = json.load(file)
                subtasks: list[SubTaskModel] = []
                for subtask_dto in json_data["subtasks"]:
                    subtask = SubTaskModel(subtask_dto["id"], subtask_dto["executionCost"], subtask_dto["memory"])
                    subtasks.append(subtask)

                dag = DAGModel(subtasks, json_data["edges"], json_data["deadline"])

            for a in range(len(algorithms)):
                algorithm = algorithms[a]

                current: float = (l * len(algorithms) * iteration + i * len(algorithms) + a)

                alg: Algorithm | None = None
                if algorithm == "Random":
                    alg = random = Random(network, dag)
                elif algorithm == "Fuzzy":
                    alg = Fuzzy(network, dag)
                elif algorithm == "NSGA3":
                    alg = NSGA3(network, dag)
                elif algorithm == "QUEST":
                    alg = QUEST(network, dag)

                alg.run()

                if i == 0:
                    result[(algorithm, load, 0)] = []
                    result[(algorithm, load, 1)] = []
                    result[(algorithm, load, 2)] = []
                    result[(algorithm, load, 3)] = []
                    result[(algorithm, load, 4)] = []

                result[(algorithm, load, 0)].append(alg.calculate_data_age())
                result[(algorithm, load, 1)].append(alg.calculate_energy())
                result[(algorithm, load, 2)].append(alg.calculate_completion_time())
                result[(algorithm, load, 3)].append(alg.calculate_load_balance())
                result[(algorithm, load, 4)].append(alg.calculate_success())

                progress((current + 1) / total)

    ws = create_sheet("Data Age")
    row = 1
    for l in range(len(loads)):
        ws.cell(row=row, column=2 + l, value=loads[l])
    for algorithm in algorithms:
        row += 1
        ws.cell(row=row, column=1, value=algorithm)
        for l in range(len(loads)):
            data_age = sum(result[(algorithm, loads[l], 0)]) / iteration
            ws.cell(row=row, column=2 + l, value=data_age)

    ws = create_sheet("Energy")
    row = 1
    for l in range(len(loads)):
        ws.cell(row=row, column=2 + l, value=loads[l])
    for algorithm in algorithms:
        row += 1
        ws.cell(row=row, column=1, value=algorithm)
        for l in range(len(loads)):
            energy = sum(result[(algorithm, loads[l], 1)]) / iteration
            ws.cell(row=row, column=2 + l, value=energy)

    ws = create_sheet("Makespan")
    row = 1
    for l in range(len(loads)):
        ws.cell(row=row, column=2 + l, value=loads[l])
    for algorithm in algorithms:
        row += 1
        ws.cell(row=row, column=1, value=algorithm)
        for l in range(len(loads)):
            makespan = sum(result[(algorithm, loads[l], 2)]) / iteration
            ws.cell(row=row, column=2 + l, value=makespan)

    ws = create_sheet("Load")
    row = 1
    for l in range(len(loads)):
        ws.cell(row=row, column=2 + l, value=loads[l])
    for algorithm in algorithms:
        row += 1
        ws.cell(row=row, column=1, value=algorithm)
        for l in range(len(loads)):
            load = sum(result[(algorithm, loads[l], 3)]) / iteration
            ws.cell(row=row, column=2 + l, value=load)

    ws = create_sheet("Success Rate")
    row = 1
    for l in range(len(loads)):
        ws.cell(row=row, column=2 + l, value=loads[l])
    for algorithm in algorithms:
        row += 1
        ws.cell(row=row, column=1, value=algorithm)
        for l in range(len(loads)):
            successrate = sum(result[(algorithm, loads[l], 4)]) / iteration
            ws.cell(row=row, column=2 + l, value=successrate)

    wb.save(output)
