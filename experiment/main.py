import configparser
import os
import subprocess
import sys

from experiment.Experiment import Experiment

if __name__ == '__main__':
    task_generator = '../workflow/TaskGenerator.py'

    config = configparser.ConfigParser()
    config.read('config.ini')

    output = config['Main']['output']
    algorithms = config['Main']['algorithms'].split(",")
    iteration = int(config['Main']['iteration'])

    env_vars = os.environ.copy()

    dag_files = {}
    loads = [int(l) for l in config['Workload']["loads"].split(",")]

    for load in loads:
        workload_config = config[f'Workload-{load}']
        for key, value in workload_config.items():
            env_vars[key.upper()] = value

        try:
            subprocess.run([sys.executable, task_generator], env=env_vars, check=True)
        except subprocess.SubprocessError as se:
            raise Exception(f"Error running the main script: {se}")

        count = int(workload_config["TOTAL_COUNT"])
        name = workload_config["OUTPUT_NAME"]
        dag_files[int(load)] = [f"./tasks/{name}/dag-{i + 1}.json" for i in range(count)]

    exp = Experiment(algorithms, loads, dag_files, iteration, output)
    exp.run()
    exp.store()

