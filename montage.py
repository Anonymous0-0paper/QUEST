from algorithms.Experiment import Experiment

mode = "montage"
loads = [50, 100, 150]
algorithms = ["Random", "Fuzzy", "NSGA3", "QUEST"]
dag_files = {
    50: [f"./workflow/Outputs/montage-50/dag-{i + 1}.json" for i in range(100)],
    100: [f"./workflow/Outputs/montage-100/dag-{i + 1}.json" for i in range(100)],
    150: [f"./workflow/Outputs/montage-150/dag-{i + 1}.json" for i in range(100)],
}
iteration = 30
output = f"./result-{mode}-1.xlsx"

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()

