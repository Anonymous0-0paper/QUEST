from algorithms.Experiment import Experiment

mode = "montage"
loads = [50, 100, 150]
algorithms = ["QUEST", "Fuzzy", "NSGA3", "MQGA", "Greedy", "MOPSO"]
dag_files = {
    50: [f"./workflow/Outputs/montage-50/dag-{i + 1}.json" for i in range(100)],
    100: [f"./workflow/Outputs/montage-100/dag-{i + 1}.json" for i in range(100)],
    150: [f"./workflow/Outputs/montage-150/dag-{i + 1}.json" for i in range(100)],
}
iteration = 100
output = f"./result-{mode}-1.xlsx"

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()

