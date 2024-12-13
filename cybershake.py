from algorithms.Experiment import Experiment

mode = "cybershake"
loads = [20, 40, 100]
algorithms = ["Random", "Fuzzy", "NSGA3", "QUEST", "MQGA", "greedy"]
dag_files = {
    20: [f"./workflow/Outputs/cybershake-20/dag-{i + 1}.json" for i in range(100)],
    40: [f"./workflow/Outputs/cybershake-40/dag-{i + 1}.json" for i in range(100)],
    100: [f"./workflow/Outputs/cybershake-100/dag-{i + 1}.json" for i in range(100)],
}
iteration = 100
output = f"./result-{mode}-1.xlsx"

if __name__ == '__main__':
    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()
