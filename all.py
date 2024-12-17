import random

from algorithms.Experiment import Experiment

mode = "all"
loads = [1, 2, 3]
algorithms = ["QUEST", "Fuzzy", "NSGA3", "MQGA", "Greedy", "MOPSO"]
iteration = 1
output = f"./result-{mode}-1.xlsx"
dag_files = {
    1: [f"./workflow/Outputs/montage-50/dag-{i + 1}.json" for i in range(100)],
    2: [f"./workflow/Outputs/montage-100/dag-{i + 1}.json" for i in range(100)],
    3: [f"./workflow/Outputs/montage-150/dag-{i + 1}.json" for i in range(100)],
}
dag_files[1].extend([f"./workflow/Outputs/cybershake-20/dag-{i + 1}.json" for i in range(100)])
dag_files[2].extend([f"./workflow/Outputs/cybershake-40/dag-{i + 1}.json" for i in range(100)])
dag_files[3].extend([f"./workflow/Outputs/cybershake-100/dag-{i + 1}.json" for i in range(100)])

dag_files[1].extend([f"./workflow/Outputs/inspiral-20/dag-{i + 1}.json" for i in range(100)])
dag_files[2].extend([f"./workflow/Outputs/inspiral-40/dag-{i + 1}.json" for i in range(100)])
dag_files[3].extend([f"./workflow/Outputs/inspiral-60/dag-{i + 1}.json" for i in range(100)])

dag_files[1].extend([f"./workflow/Outputs/epigenomics-20/dag-{i + 1}.json" for i in range(100)])
dag_files[2].extend([f"./workflow/Outputs/epigenomics-30/dag-{i + 1}.json" for i in range(100)])
dag_files[3].extend([f"./workflow/Outputs/epigenomics-50/dag-{i + 1}.json" for i in range(100)])

random.shuffle(dag_files[1])
random.shuffle(dag_files[2])
random.shuffle(dag_files[3])

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()

