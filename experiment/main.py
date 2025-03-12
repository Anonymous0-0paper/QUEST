# import configparser
# import os
# import subprocess
# import sys
#
# from experiment.Experiment import Experiment
#
# if __name__ == '__main__':
#     task_generator = '../workflow/TaskGenerator.py'
#
#     # Create the ConfigParser and preserve key case
#     config = configparser.ConfigParser()
#     config.optionxform = str  # Preserve keys as they are (do not lowercase them)
#
#     # Define the configuration file path
#     config_file_path = '/home/user/PycharmProjects/QUESTnew/experiment/montage/config.ini'
#
#     # Read the configuration file
#     files_read = config.read(config_file_path)
#     if not files_read:
#         raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
#
#     # Now the sections should be available.
#     output = config.get('Main', 'output')
#     algorithms = [algo.strip() for algo in config.get('Main', 'algorithms').split(",")]
#     iteration = int(config.get('Main', 'iteration'))
#
#     env_vars = os.environ.copy()
#     dag_files = {}
#     loads = [int(l) for l in config.get('Workload', 'loads').split(",")]
#
#     for load in loads:
#         workload_config = config[f'Workload-{load}']
#         # Update environment variables with workload settings (preserving key case)
#         for key, value in workload_config.items():
#             env_vars[key.upper()] = value
#
#         try:
#             subprocess.run([sys.executable, task_generator], env=env_vars, check=True)
#         except subprocess.SubprocessError as se:
#             raise Exception(f"Error running the main script: {se}")
#
#         count = int(workload_config["TOTAL_COUNT"])
#         name = workload_config["OUTPUT_NAME"]
#         dag_files[load] = [f"./tasks/{name}/dag-{i + 1}.json" for i in range(count)]
#
#     exp = Experiment(algorithms, loads, dag_files, iteration, output)
#     exp.run()
#     exp.store()

import os
import sys
import subprocess
import glob
import configparser
import concurrent.futures
from experiment.Experiment import Experiment

def process_ini_file(ini_file):
    """Process a single INI configuration file."""
    print(f"Processing configuration: {ini_file}")

    # Load the configuration file and preserve key case.
    config = configparser.ConfigParser()
    config.optionxform = str  # Do not convert keys to lowercase.
    config.read(ini_file)

    try:
        output = config['Main']['output']
        algorithms = [algo.strip() for algo in config['Main']['algorithms'].split(",")]
        iteration = int(config['Main']['iteration'])
    except KeyError as e:
        raise Exception(f"Missing key in [Main] section in {ini_file}: {e}")

    # Prepare environment variables and collect DAG file paths.
    env_vars = os.environ.copy()
    dag_files = {}

    try:
        loads = [int(l.strip()) for l in config['Workload']["loads"].split(",")]
    except KeyError as e:
        raise Exception(f"Missing key in [Workload] section in {ini_file}: {e}")

    for load in loads:
        section_name = f'Workload-{load}'
        if section_name not in config:
            print(f"Section {section_name} not found in {ini_file}. Skipping load {load}.")
            continue

        workload_config = config[section_name]
        # Update environment variables with workload-specific values.
        for key, value in workload_config.items():
            env_vars[key.upper()] = value

        # Run the task generator.
        try:
            subprocess.run([sys.executable, '../workflow/TaskGenerator.py'],
                           env=env_vars, check=True)
        except subprocess.SubprocessError as se:
            raise Exception(f"Error running TaskGenerator for {ini_file} on load {load}: {se}")

        try:
            count = int(workload_config["TOTAL_COUNT"])
            name = workload_config["OUTPUT_NAME"]
        except KeyError as e:
            raise Exception(f"Missing key in {section_name} of {ini_file}: {e}")

        # Collect DAG file paths.
        dag_files[load] = [f"./tasks/{name}/dag-{i + 1}.json" for i in range(count)]

    # Create and run the experiment.
    try:
        exp = Experiment(algorithms, loads, dag_files, iteration, output,ini_file)
        exp.run()
        exp.store()
    except Exception as e:
        raise Exception(f"Error running experiment for {ini_file}: {e}")

    return f"Finished processing configuration: {ini_file}"

def process_directory(ini_dir):
    """Process all INI files in a given directory with up to 3 workers concurrently."""
    print(f"Processing directory: {ini_dir}")
    ini_files = glob.glob(os.path.join(ini_dir, "*.ini"))

    if not ini_files:
        print(f"No INI configuration files found in {ini_dir}.")
        return []

    results = []
    # Create a ProcessPoolExecutor with 3 workers for this directory.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        future_to_ini = {executor.submit(process_ini_file, ini_file): ini_file for ini_file in ini_files}
        for future in concurrent.futures.as_completed(future_to_ini):
            ini_file = future_to_ini[future]
            try:
                result = future.result()
                print(result)
                results.append(result)
            except Exception as exc:
                print(f"{ini_file} generated an exception: {exc}")
    return results

if __name__ == '__main__':
    # List of directories containing INI files.
    ini_dirs = [
        # "/home/user/PycharmProjects/QUESTnew/experiment/montage",
        # "/home/user/PycharmProjects/QUESTnew/experiment/inspiral",
        # "/home/user/PycharmProjects/QUESTnew/experiment/cyber",
        "/home/user/PycharmProjects/QUESTnew/experiment/epigenomics"
    ]

    # Process each directory concurrently.
    # Using a ThreadPoolExecutor here since each directory spawns its own ProcessPoolExecutor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ini_dirs)) as dir_executor:
        futures = {dir_executor.submit(process_directory, ini_dir): ini_dir for ini_dir in ini_dirs}
        for future in concurrent.futures.as_completed(futures):
            ini_dir = futures[future]
            try:
                _ = future.result()
                print(f"Finished processing directory: {ini_dir}")
            except Exception as exc:
                print(f"{ini_dir} generated an exception: {exc}")
