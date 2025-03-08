import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

from workflow.DAG import DAG, DAGMode
from workflow.SubTask import SubTask


class Pegasus:
    """
    Class for parsing Pegasus DAX XML files and generating DAG objects.
    """

    @staticmethod
    def parse_xml(xml_path: str) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract namespace from root tag (if present)
        ns = {'dax': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

        # Find the correct tag prefix for elements
        if ns:
            job_tag = f"{{{ns['dax']}}}job"
            child_tag = f"{{{ns['dax']}}}child"
            parent_tag = f"{{{ns['dax']}}}parent"
            uses_tag = f"{{{ns['dax']}}}uses"
        else:
            job_tag = "job"
            child_tag = "child"
            parent_tag = "parent"
            uses_tag = "uses"

        # Extract job information
        jobs = {}
        for job in root.findall(f".//{job_tag}"):
            job_id = job.attrib.get('id')
            namespace = job.attrib.get('namespace', '')
            name = job.attrib.get('name', '')
            version = job.attrib.get('version', '')
            runtime = float(job.attrib.get('runtime', 0))

            # Get file information
            files = []
            input_size = 0
            output_size = 0

            for file_info in job.findall(f".//{uses_tag}"):
                file_name = file_info.attrib.get('file', '')
                link_type = file_info.attrib.get('link', '')
                register = file_info.attrib.get('register', 'false').lower() == 'true'
                transfer = file_info.attrib.get('transfer', 'false').lower() == 'true'
                optional = file_info.attrib.get('optional', 'false').lower() == 'true'
                file_type = file_info.attrib.get('type', '')
                size = int(file_info.attrib.get('size', 0))

                file_data = {
                    'name': file_name,
                    'link': link_type,
                    'register': register,
                    'transfer': transfer,
                    'optional': optional,
                    'type': file_type,
                    'size': size
                }

                files.append(file_data)

                # Track input and output sizes
                if link_type.lower() == 'input':
                    input_size += size
                elif link_type.lower() == 'output':
                    output_size += size

            jobs[job_id] = {
                'id': job_id,
                'namespace': namespace,
                'name': name,
                'version': version,
                'runtime': runtime,
                'files': files,
                'input_size': input_size,
                'output_size': output_size
            }

        # Extract dependencies (parent-child relationships)
        dependencies = {}
        for child_elem in root.findall(f".//{child_tag}"):
            child_id = child_elem.attrib.get('ref')
            parents = []

            for parent_elem in child_elem.findall(f".//{parent_tag}"):
                parent_id = parent_elem.attrib.get('ref')
                parents.append(parent_id)

            dependencies[child_id] = parents

        return jobs, dependencies

    @staticmethod
    def generate_dag(dag_id: int, xml_path: str, deadline_min: int, deadline_max: int, mips: int) -> DAG:

        jobs, dependencies = Pegasus.parse_xml(xml_path)

        # Create a mapping from job IDs to numerical IDs
        job_to_id = {job_id: idx for idx, job_id in enumerate(jobs.keys())}

        tasks = []
        for job_id, job_info in jobs.items():
            runtime_seconds = max(0.1, job_info['runtime'])  # Ensure at least 0.1 second
            execution_cost = int(runtime_seconds * mips)  # Convert seconds to MI

            io_factor = (job_info['input_size'] + job_info['output_size']) / 1_000_000_000

            task = SubTask(dag_id, job_to_id[job_id])
            task.memory = io_factor
            task.execution_cost = execution_cost
            tasks.append(task)

        edges = []
        for child_id, parent_ids in dependencies.items():
            if child_id in job_to_id:
                child_idx = job_to_id[child_id]
                child_job = jobs[child_id]

                for parent_id in parent_ids:
                    if parent_id in job_to_id:
                        parent_idx = job_to_id[parent_id]
                        parent_job = jobs[parent_id]

                        # Calculate data transfer size based on shared files
                        data_size = 0

                        # Find output files from parent
                        parent_output_files = {file_info['name']: file_info['size']
                                               for file_info in parent_job['files']
                                               if file_info['link'].lower() == 'output'}

                        # Find input files to child that match parent's output
                        for file_info in child_job['files']:
                            if file_info['link'].lower() == 'input' and file_info['name'] in parent_output_files:
                                data_size += file_info['size']

                        # Apply scaling (GB)
                        data_size = data_size / 1_000_000_000

                        edges.append([parent_idx, child_idx, data_size])

        # Create DAG
        dag = DAG(DAGMode.PegasusWorkflow, dag_id, tasks, edges)
        dag.add_dummy_entry()
        dag.generate_deadline(deadline_min, deadline_max)

        return dag