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
import logging
from typing import Dict, List, Any
from experiment.Experiment import Experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_ini_file(ini_file: str) -> str:
    """
    Process a single INI configuration file with error handling.
    
    Args:
        ini_file (str): Path to the INI configuration file
        
    Returns:
        str: Status message indicating completion
        
    Raises:
        FileNotFoundError: If the INI file doesn't exist
        configparser.Error: If the INI file is invalid
        KeyError: If required configuration keys are missing
        subprocess.SubprocessError: If task generation fails
        Exception: For other unexpected errors
    """
    try:
        if not os.path.exists(ini_file):
            raise FileNotFoundError(f"Configuration file not found: {ini_file}")
            
        logger.info(f"Processing configuration: {ini_file}")

        # Load the configuration file and preserve key case
        config = configparser.ConfigParser()
        config.optionxform = str  # Do not convert keys to lowercase
        config.read(ini_file)

        # Validate and extract main configuration
        try:
            output = config['Main']['output']
            algorithms = [algo.strip() for algo in config['Main']['algorithms'].split(",")]
            iteration = int(config['Main']['iteration'])
        except KeyError as e:
            raise KeyError(f"Missing key in [Main] section in {ini_file}: {e}")

        # Prepare environment variables and collect DAG file paths
        env_vars = os.environ.copy()
        dag_files: Dict[int, List[str]] = {}

        # Validate and extract workload configuration
        try:
            loads = [int(l.strip()) for l in config['Workload']["loads"].split(",")]
        except KeyError as e:
            raise KeyError(f"Missing key in [Workload] section in {ini_file}: {e}")

        # Process each load configuration
        for load in loads:
            section_name = f'Workload-{load}'
            if section_name not in config:
                logger.warning(f"Section {section_name} not found in {ini_file}. Skipping load {load}.")
                continue

            workload_config = config[section_name]
            
            # Update environment variables with workload-specific values
            for key, value in workload_config.items():
                env_vars[key.upper()] = value

            # Run the task generator
            try:
                logger.info(f"Running TaskGenerator for load {load}")
                subprocess.run([sys.executable, '../workflow/TaskGenerator.py'],
                             env=env_vars, check=True)
                logger.info(f"TaskGenerator completed successfully for load {load}")
            except subprocess.SubprocessError as se:
                raise subprocess.SubprocessError(f"Error running TaskGenerator for {ini_file} on load {load}: {se}")

            # Validate and extract workload-specific configuration
            try:
                count = int(workload_config["TOTAL_COUNT"])
                name = workload_config["OUTPUT_NAME"]
            except KeyError as e:
                raise KeyError(f"Missing key in {section_name} of {ini_file}: {e}")

            # Collect DAG file paths
            dag_files[load] = [f"./tasks/{name}/dag-{i + 1}.json" for i in range(count)]
            logger.info(f"Collected {count} DAG files for load {load}")

        # Create and run the experiment
        try:
            logger.info(f"Creating experiment with {len(algorithms)} algorithms and {len(loads)} loads")
            exp = Experiment(algorithms, loads, dag_files, iteration, output, ini_file)
            exp.run()
            exp.store()
            logger.info(f"Experiment completed and results stored successfully")
        except Exception as e:
            raise Exception(f"Error running experiment for {ini_file}: {e}")

        return f"Finished processing configuration: {ini_file}"
        
    except Exception as e:
        logger.error(f"Error processing configuration file {ini_file}: {str(e)}")
        raise

def process_directory(ini_dir: str) -> List[str]:
    """
    Process all INI files in a given directory with concurrent execution.
    
    Args:
        ini_dir (str): Directory containing INI configuration files
        
    Returns:
        List[str]: List of processing results
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        Exception: For other unexpected errors
    """
    try:
        if not os.path.exists(ini_dir):
            raise FileNotFoundError(f"Directory not found: {ini_dir}")
            
        logger.info(f"Processing directory: {ini_dir}")
        ini_files = glob.glob(os.path.join(ini_dir, "*.ini"))

        if not ini_files:
            logger.warning(f"No INI configuration files found in {ini_dir}.")
            return []

        results = []
        # Create a ProcessPoolExecutor with 3 workers for this directory
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            future_to_ini = {executor.submit(process_ini_file, ini_file): ini_file 
                           for ini_file in ini_files}
            
            for future in concurrent.futures.as_completed(future_to_ini):
                ini_file = future_to_ini[future]
                try:
                    result = future.result()
                    logger.info(result)
                    results.append(result)
                except Exception as exc:
                    logger.error(f"{ini_file} generated an exception: {exc}")
                    
        return results
        
    except Exception as e:
        logger.error(f"Error processing directory {ini_dir}: {str(e)}")
        raise

def main():
    """
    Main function to run the experiment processing with concurrent execution.
    """
    try:
        logger.info("Starting experiment processing")
        
        # List of directories containing INI files
        ini_dirs = [
            "/home/user/PycharmProjects/QUESTnew/experiment/scientific1000"
        ]

        # Process each directory concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ini_dirs)) as dir_executor:
            futures = {dir_executor.submit(process_directory, ini_dir): ini_dir 
                      for ini_dir in ini_dirs}
            
            for future in concurrent.futures.as_completed(futures):
                ini_dir = futures[future]
                try:
                    _ = future.result()
                    logger.info(f"Finished processing directory: {ini_dir}")
                except Exception as exc:
                    logger.error(f"{ini_dir} generated an exception: {exc}")
                    
        logger.info("Experiment processing completed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()
