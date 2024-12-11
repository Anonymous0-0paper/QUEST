from algorithms.Experiment import Experiment

mode = "inspiral"
loads = [20, 40, 60]
algorithms = ["Random", "Fuzzy", "NSGA3", "QUEST"]
dag_files = {
    20: [f"./workflow/Outputs/inspiral-20/dag-{i + 1}.json" for i in range(100)],
    40: [f"./workflow/Outputs/inspiral-40/dag-{i + 1}.json" for i in range(100)],
    60: [f"./workflow/Outputs/inspiral-60/dag-{i + 1}.json" for i in range(100)],
}
iteration = 10
output = f"./result-{mode}.xlsx"

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
