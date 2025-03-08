import configparser
import os
import pandas as pd
import json
import shutil
import logging
import datetime
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

class ConfigValidator:
    """Validator class to ensure configuration values are within acceptable ranges."""

    # Define valid ranges and constraints for different parameters
    VALID_RANGES = {
        'node_count': (1, 100),  # Min and max node count
        'cpu_count': (1, 128),   # Min and max CPU count
        'ram': (1, 1024),        # Min and max RAM in GB
        'dag_size': (10, 1000),  # Min and max DAG size
        'total_count': (1, 1000) # Min and max total count
    }

    VALID_DAG_TYPES = [
        'CaseWind', 'Montage', 'Genome1000', 'CyberShake',
        'Epigenomics', 'Inspiral', 'Random', 'FullTopology'
    ]

    @staticmethod
    def validate_int_range(value: int, param_type: str) -> Tuple[bool, str]:
        """Validate if an integer value is within the acceptable range."""
        if param_type not in ConfigValidator.VALID_RANGES:
            return False, f"Unknown parameter type: {param_type}"

        min_val, max_val = ConfigValidator.VALID_RANGES[param_type]
        if min_val <= value <= max_val:
            return True, ""
        else:
            return False, f"Value {value} for {param_type} is outside acceptable range ({min_val}-{max_val})"

    @staticmethod
    def validate_dag_type(dag_type: str) -> Tuple[bool, str]:
        """Validate if the DAG type is valid."""
        if dag_type in ConfigValidator.VALID_DAG_TYPES:
            return True, ""
        else:
            return False, f"Invalid DAG type: {dag_type}. Valid types are: {', '.join(ConfigValidator.VALID_DAG_TYPES)}"

    @staticmethod
    def validate_node_type(node_type: str) -> Tuple[bool, str]:
        """Validate if the node type is valid."""
        if node_type.lower() in ['edge', 'cloud']:
            return True, ""
        else:
            return False, f"Invalid node type: {node_type}. Valid types are: Edge, Cloud"


class ConfigModifier:
    """Class to read, modify, and save configuration files for simulation experiments."""

    def __init__(self, config_path: str, log_level: int = logging.INFO):
        """
        Initialize the ConfigModifier with the path to the config file.

        Args:
            config_path (str): Path to the configuration file
            log_level (int): Logging level (default: logging.INFO)
        """
        self.config_path = config_path
        self.config = configparser.ConfigParser(allow_no_value=True, comment_prefixes=('#',))
        # Preserve case sensitivity
        self.config.optionxform = str

        # Set up logging
        self.setup_logging(log_level)

        # Try to read the config file
        try:
            self.config.read(config_path)
            self.logger.info(f"Successfully loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

        # Create backup of original configuration
        self.create_backup()

    def setup_logging(self, log_level: int) -> None:
        """Set up logging for the ConfigModifier."""
        self.logger = logging.getLogger('ConfigModifier')
        self.logger.setLevel(log_level)

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create a file handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"config_modifier_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)

        # Create a console handler
        console_handler = logging.StreamHandler()

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def create_backup(self) -> str:
        """
        Create a backup of the current configuration file.

        Returns:
            str: Path to the backup file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)

        backup_filename = f"{Path(self.config_path).stem}_{timestamp}.bak"
        backup_path = backup_dir / backup_filename

        try:
            shutil.copy2(self.config_path, backup_path)
            self.logger.info(f"Created backup at {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            return ""

    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore configuration from a backup file.

        Args:
            backup_path (str): Path to the backup file

        Returns:
            bool: True if restoration was successful, False otherwise
        """
        try:
            shutil.copy2(backup_path, self.config_path)
            # Reload the configuration
            self.config.read(self.config_path)
            self.logger.info(f"Restored configuration from {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {str(e)}")
            return False

    def list_backups(self) -> List[str]:
        """
        List all available backup files.

        Returns:
            List[str]: List of backup file paths
        """
        backup_dir = Path("backups")
        if not backup_dir.exists():
            return []

        # Find all backups for the current config file
        config_name = Path(self.config_path).stem
        backups = list(backup_dir.glob(f"{config_name}_*.bak"))
        return [str(b) for b in backups]

    def modify_edge_cloud_nodes(self, region: str, node_type: str, new_count: int) -> bool:
        """
        Modify the number of nodes for a specific region and type (Edge or Cloud).

        Args:
            region (str): The region name (e.g., 'lyon', 'luxembourg', 'toulouse')
            node_type (str): The node type ('edge' or 'cloud')
            new_count (int): New number of nodes

        Returns:
            bool: True if modification was successful, False otherwise
        """
        node_type = node_type.lower()
        region = region.lower()

        # Validate inputs
        valid, message = ConfigValidator.validate_node_type(node_type)
        if not valid:
            self.logger.error(message)
            return False



        # Check if the node type exists for the region
        region_types_key = f"{region}_types"
        if region_types_key in self.config['Nodes']:
            region_types = self.config['Nodes'][region_types_key].split(', ')

            # Check if the node type is valid for this region
            if node_type.capitalize() in region_types:
                count_key = f"{region}_{node_type}_count"
                self.config['Nodes'][count_key] = str(new_count)
                self.logger.info(f"Updated {region} {node_type} count to {new_count}")
                return True
            else:
                self.logger.error(f"Error: {node_type.capitalize()} is not a valid node type for {region}.")
                return False
        else:
            self.logger.error(f"Error: Region '{region}' not found in configuration.")
            return False

    def modify_dag_size(self, workload_id: Union[int, str], new_size: int) -> bool:
        """
        Modify the DAG size for a specific workload.

        Args:
            workload_id (int or str): The workload ID (e.g., 1, 2, 3)
            new_size (int): New DAG size

        Returns:
            bool: True if modification was successful, False otherwise
        """
        workload_section = f"Workload-{workload_id}"

        # Validate input
        valid, message = ConfigValidator.validate_int_range(new_size, 'dag_size')
        if not valid:
            self.logger.error(message)
            return False

        if workload_section in self.config:
            self.config[workload_section]['DAG_SIZE'] = str(new_size)
            self.logger.info(f"Updated {workload_section} DAG_SIZE to {new_size}")
            return True
        else:
            self.logger.error(f"Error: Workload '{workload_id}' not found in configuration.")
            return False

    def modify_dag_type(self, workload_id: Union[int, str], new_type: str) -> bool:
        """
        Modify the DAG type for a specific workload.

        Args:
            workload_id (int or str): The workload ID (e.g., 1, 2, 3)
            new_type (str): New DAG type

        Returns:
            bool: True if modification was successful, False otherwise
        """
        workload_section = f"Workload-{workload_id}"

        # Validate input
        valid, message = ConfigValidator.validate_dag_type(new_type)
        if not valid:
            self.logger.error(message)
            return False

        if workload_section in self.config:
            self.config[workload_section]['DAG_TYPE'] = new_type
            self.logger.info(f"Updated {workload_section} DAG_TYPE to {new_type}")
            return True
        else:
            self.logger.error(f"Error: Workload '{workload_id}' not found in configuration.")
            return False

    def save_config(self, output_path: Optional[str] = None) -> bool:
        """
        Save the modified configuration to a file.

        Args:
            output_path (str, optional): Path to save the modified config file.
                                        If None, original file will be overwritten.

        Returns:
            bool: True if save was successful, False otherwise
        """
        if output_path is None:
            output_path = self.config_path

        try:
            with open(output_path, 'w') as configfile:
                self.config.write(configfile)

            self.logger.info(f"Configuration saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            return False

    def create_output_structure(self) -> Dict[str, str]:
        """
        Create the output folder structure and Excel file based on the configuration.

        Returns:
            Dict[str, str]: Dictionary with paths to created files
        """
        created_files = {}

        try:
            output_path = self.config.get('Main', 'output', fallback='../results.xlsx')

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Create Excel file with multiple sheets
            if not os.path.exists(output_path):
                # Create a writer object
                writer = pd.ExcelWriter(output_path, engine='openpyxl')

                # Create Configuration sheet
                self._create_config_sheet(writer)

                # Create Nodes sheet
                self._create_nodes_sheet(writer)

                # Create Workloads sheet
                self._create_workloads_sheet(writer)

                # Create Regions sheet
                self._create_regions_sheet(writer)

                # Save the Excel file
                writer.close()

                self.logger.info(f"Created output Excel file at {output_path}")
                created_files['excel'] = output_path
            else:
                self.logger.info(f"Output Excel file already exists at {output_path}")
                created_files['excel'] = output_path

            # Create workload output directories
            workload_ids = self.config.get('Workload', 'loads', fallback='').split(',')
            for wl_id in workload_ids:
                wl_id = wl_id.strip()
                if f'Workload-{wl_id}' in self.config:
                    output_name = self.config.get(f'Workload-{wl_id}', 'OUTPUT_NAME', fallback=f'workload-{wl_id}')
                    workload_dir = Path(output_dir) / output_name
                    workload_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created workload directory at {workload_dir}")
                    created_files[f'workload-{wl_id}'] = str(workload_dir)

            return created_files
        except Exception as e:
            self.logger.error(f"Failed to create output structure: {str(e)}")
            return created_files

    def _create_config_sheet(self, writer: pd.ExcelWriter) -> None:
        """Create the Configuration sheet in the Excel file."""
        # Create a basic DataFrame with information from the config
        data = {
            'Parameter': ['Configuration File', 'Algorithms', 'Iterations', 'Regions'],
            'Value': [
                self.config_path,
                self.config.get('Main', 'algorithms', fallback=''),
                self.config.get('Main', 'iteration', fallback=''),
                self.config.get('Regions', 'region_names', fallback='')
            ]
        }

        # Add information about workloads
        workload_ids = self.config.get('Workload', 'loads', fallback='').split(',')
        for wl_id in workload_ids:
            wl_id = wl_id.strip()
            if f'Workload-{wl_id}' in self.config:
                data['Parameter'].append(f'Workload {wl_id} - DAG Type')
                data['Value'].append(self.config.get(f'Workload-{wl_id}', 'DAG_TYPE', fallback='N/A'))

                data['Parameter'].append(f'Workload {wl_id} - DAG Size')
                data['Value'].append(self.config.get(f'Workload-{wl_id}', 'DAG_SIZE', fallback='N/A'))

                data['Parameter'].append(f'Workload {wl_id} - Total Count')
                data['Value'].append(self.config.get(f'Workload-{wl_id}', 'TOTAL_COUNT', fallback='N/A'))

        # Create a DataFrame and save to Excel
        df = pd.DataFrame(data)
        df.to_excel(writer, index=False, sheet_name='Configuration')

    def _create_nodes_sheet(self, writer: pd.ExcelWriter) -> None:
        """Create the Nodes sheet in the Excel file."""
        # Get all regions
        regions = self.config.get('Regions', 'region_names', fallback='').split(', ')

        # Create a list to store node data
        node_data = []

        for region in regions:
            region_lower = region.lower()
            region_types_key = f"{region_lower}_types"

            if region_types_key in self.config['Nodes']:
                node_types = self.config['Nodes'][region_types_key].split(', ')

                for node_type in node_types:
                    node_type_lower = node_type.lower()
                    prefix = f"{region_lower}_{node_type_lower}"

                    # Get node properties
                    server_model = self.config['Nodes'].get(f"{prefix}_server_model", "N/A")
                    cpu_count = self.config['Nodes'].get(f"{prefix}_cpu_count", "N/A")
                    cpu_speed = self.config['Nodes'].get(f"{prefix}_cpu_speed", "N/A")
                    ram = self.config['Nodes'].get(f"{prefix}_ram", "N/A")
                    min_load = self.config['Nodes'].get(f"{prefix}_min_load", "N/A")
                    max_load = self.config['Nodes'].get(f"{prefix}_max_load", "N/A")
                    count = self.config['Nodes'].get(f"{prefix}_count", "N/A")

                    # Add to node data
                    node_data.append({
                        'Region': region,
                        'Type': node_type,
                        'Server Model': server_model,
                        'CPU Count': cpu_count,
                        'CPU Speed': cpu_speed,
                        'RAM (GB)': ram,
                        'Min Load (%)': min_load,
                        'Max Load (%)': max_load,
                        'Node Count': count
                    })

        # Create a DataFrame and save to Excel
        if node_data:
            df = pd.DataFrame(node_data)
            df.to_excel(writer, index=False, sheet_name='Nodes')

    def _create_workloads_sheet(self, writer: pd.ExcelWriter) -> None:
        """Create the Workloads sheet in the Excel file."""
        # Get all workload IDs
        workload_ids = self.config.get('Workload', 'loads', fallback='').split(',')

        # Create a list to store workload data
        workload_data = []

        for wl_id in workload_ids:
            wl_id = wl_id.strip()
            workload_section = f"Workload-{wl_id}"

            if workload_section in self.config:
                # Get workload properties
                output_name = self.config[workload_section].get('OUTPUT_NAME', f"workload-{wl_id}")
                dag_type = self.config[workload_section].get('DAG_TYPE', "N/A")
                dag_size = self.config[workload_section].get('DAG_SIZE', "N/A")
                total_count = self.config[workload_section].get('TOTAL_COUNT', "N/A")

                # Add to workload data
                workload_data.append({
                    'Workload ID': wl_id,
                    'Output Name': output_name,
                    'DAG Type': dag_type,
                    'DAG Size': dag_size,
                    'Total Count': total_count,
                    'Comm Min': self.config[workload_section].get('DAG_COMMUNICATION_MIN', "N/A"),
                    'Comm Max': self.config[workload_section].get('DAG_COMMUNICATION_MAX', "N/A"),
                    'Comp Min': self.config[workload_section].get('DAG_COMPUTATION_MIN', "N/A"),
                    'Comp Max': self.config[workload_section].get('DAG_COMPUTATION_MAX', "N/A"),
                    'Deadline Min': self.config[workload_section].get('DAG_DEADLINE_MIN', "N/A"),
                    'Deadline Max': self.config[workload_section].get('DAG_DEADLINE_MAX', "N/A"),
                    'Memory Min': self.config[workload_section].get('DAG_MEMORY_MIN', "N/A"),
                    'Memory Max': self.config[workload_section].get('DAG_MEMORY_MAX', "N/A")
                })

        # Create a DataFrame and save to Excel
        if workload_data:
            df = pd.DataFrame(workload_data)
            df.to_excel(writer, index=False, sheet_name='Workloads')

    def _create_regions_sheet(self, writer: pd.ExcelWriter) -> None:
        """Create the Regions sheet in the Excel file."""
        # Get all regions
        regions = self.config.get('Regions', 'region_names', fallback='').split(', ')

        # Create a list to store connection data
        connection_data = []

        # Process connections
        for key, value in self.config['Connections'].items():
            # Parse connection key
            match = re.match(r"([a-z]+)_([a-z]+)_([a-z_]+)", key)
            if match:
                region1, region2, metric = match.groups()

                # Add to connection data
                connection_data.append({
                    'Region 1': region1,
                    'Region 2': region2,
                    'Metric': metric,
                    'Value': value
                })

        # Create a DataFrame and save to Excel
        if connection_data:
            df = pd.DataFrame(connection_data)
            df.to_excel(writer, index=False, sheet_name='Connections')

    def get_all_node_info(self) -> Dict[str, Dict[str, str]]:
        """
        Return a dictionary with all node information for all regions.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary with node information
        """
        node_info = {}

        regions = self.config.get('Regions', 'region_names', fallback='').split(', ')

        for region in regions:
            region_lower = region.lower()
            region_types_key = f"{region_lower}_types"

            if region_types_key in self.config['Nodes']:
                node_types = self.config['Nodes'][region_types_key].split(', ')

                node_info[region] = {}

                for node_type in node_types:
                    node_type_lower = node_type.lower()
                    count_key = f"{region_lower}_{node_type_lower}_count"

                    if count_key in self.config['Nodes']:
                        node_count = self.config['Nodes'][count_key]
                        node_info[region][node_type] = node_count

        return node_info

    def get_all_dag_sizes(self) -> Dict[str, str]:
        """
        Return a dictionary with all DAG sizes for all workloads.

        Returns:
            Dict[str, str]: Dictionary with DAG sizes
        """
        dag_sizes = {}

        workload_ids = self.config.get('Workload', 'loads', fallback='').split(',')

        for wl_id in workload_ids:
            wl_id = wl_id.strip()
            workload_section = f"Workload-{wl_id}"

            if workload_section in self.config:
                dag_size = self.config[workload_section].get('DAG_SIZE', 'N/A')
                dag_sizes[wl_id] = dag_size

        return dag_sizes

    def export_config_to_json(self, json_path: Optional[str] = None) -> str:
        """
        Export the current configuration to a JSON file.

        Args:
            json_path (str, optional): Path to save the JSON file.
                                      If None, a default path will be used.

        Returns:
            str: Path to the saved JSON file
        """
        if json_path is None:
            # Create a default path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"{Path(self.config_path).stem}_{timestamp}.json"

        # Convert ConfigParser to dictionary
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = dict(self.config[section])

        try:
            with open(json_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

            self.logger.info(f"Exported configuration to JSON at {json_path}")
            return json_path
        except Exception as e:
            self.logger.error(f"Failed to export configuration to JSON: {str(e)}")
            return ""

    def import_config_from_json(self, json_path: str) -> bool:
        """
        Import configuration from a JSON file.

        Args:
            json_path (str): Path to the JSON file

        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)

            # Create a new ConfigParser
            new_config = configparser.ConfigParser(allow_no_value=True, comment_prefixes=('#',))
            new_config.optionxform = str

            # Add sections and options from the dictionary
            for section, options in config_dict.items():
                if not new_config.has_section(section):
                    new_config.add_section(section)

                for option, value in options.items():
                    new_config[section][option] = value

            # Replace the current config with the new one
            self.config = new_config

            self.logger.info(f"Imported configuration from JSON at {json_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to import configuration from JSON: {str(e)}")
            return False

    def create_config_template(self, template_name: str) -> str:
        """
        Create a configuration template from the current configuration.

        Args:
            template_name (str): Name for the template

        Returns:
            str: Path to the saved template
        """
        # Create templates directory if it doesn't exist
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)

        template_path = templates_dir / f"{template_name}.ini"

        try:
            # Save the current config as a template
            with open(template_path, 'w') as configfile:
                self.config.write(configfile)

            self.logger.info(f"Created configuration template at {template_path}")
            return str(template_path)
        except Exception as e:
            self.logger.error(f"Failed to create configuration template: {str(e)}")
            return ""

    def generate_config_from_template(self, template_path: str, output_path: str,
                                      modifications: Optional[Dict[str, Dict[str, str]]] = None) -> bool:
        """
        Generate a new configuration file from a template with optional modifications.

        Args:
            template_path (str): Path to the template file
            output_path (str): Path for the new configuration file
            modifications (Dict[str, Dict[str, str]], optional): Dictionary of modifications
                                                              to apply to the template

        Returns:
            bool: True if generation was successful, False otherwise
        """
        try:
            # Create a new ConfigParser
            new_config = configparser.ConfigParser(allow_no_value=True, comment_prefixes=('#',))
            new_config.optionxform = str

            # Read the template
            new_config.read(template_path)

            # Apply modifications if provided
            if modifications:
                for section, options in modifications.items():
                    if not new_config.has_section(section):
                        new_config.add_section(section)

                    for option, value in options.items():
                        new_config[section][option] = value

            # Save the new configuration
            with open(output_path, 'w') as configfile:
                new_config.write(configfile)

            self.logger.info(f"Generated configuration from template at {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate configuration from template: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Create an instance of ConfigModifier with the path to your config file
        config_path = "config.ini"  # Update this to the actual path of your config file
        modifier = ConfigModifier(config_path)

        # Print current node and DAG information
        logger.info("Current Node Configuration:")
        for region, types in modifier.get_all_node_info().items():
            for node_type, count in types.items():
                logger.info(f"  {region} {node_type}: {count}")

        logger.info("\nCurrent DAG Sizes:")
        for workload_id, size in modifier.get_all_dag_sizes().items():
            logger.info(f"  Workload-{workload_id}: {size}")

        # Example modifications
        logger.info("\nModifying configuration...")

        # Modify edge nodes for Lyon
        modifier.modify_edge_cloud_nodes("lyon", "edge", 10)

        # Modify cloud nodes for Toulouse
        modifier.modify_edge_cloud_nodes("toulouse", "cloud", 3)

        # Modify DAG size for Workload-1
        modifier.modify_dag_size(1, 75)

        # Create a configuration template
        template_path = modifier.create_config_template("base_config")

        # Generate a new configuration from the template with modifications
        modifications = {
            'Nodes': {
                'lyon_edge_count': '15',
                'toulouse_cloud_count': '5'
            },
            'Workload-1': {
                'DAG_SIZE': '150'
            }
        }
        modifier.generate_config_from_template(template_path, "new_generated_config.ini", modifications)

        # Export configuration to JSON
        json_path = modifier.export_config_to_json()

        # Save the modified configuration
        modifier.save_config("New_config.ini")

        # Create output structure
        created_files = modifier.create_output_structure()
        logger.info(f"Created files: {created_files}")

        logger.info("\nDone!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")