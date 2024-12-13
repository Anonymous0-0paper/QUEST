from algorithms.Experiment import Experiment

mode = "epigenomics"
loads = [20, 30, 50]
algorithms = ["QUEST", "Fuzzy", "NSGA3", "MQGA", "Greedy", "MOPSO"]
dag_files = {
    20: [f"./workflow/Outputs/epigenomics-20/dag-{i + 1}.json" for i in range(100)],
    30: [f"./workflow/Outputs/epigenomics-30/dag-{i + 1}.json" for i in range(100)],
    50: [f"./workflow/Outputs/epigenomics-50/dag-{i + 1}.json" for i in range(100)],
}
iteration = 100
output = f"./result-{mode}-1.xlsx"

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()
