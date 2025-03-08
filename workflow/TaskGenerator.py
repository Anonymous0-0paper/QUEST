import os
import random

from workflow.DAG import DAG
from workflow.Models.Pegasus import Pegasus
from workflow.Models.CasaWind import CaseWind
from workflow.Models.CyberShake import CyberShake
from workflow.Models.Epigenomics import Epigenomics
from workflow.Models.FullTopology import FullTopology
from workflow.Models.Genome1000 import Genome1000
from workflow.Models.Inspiral import Inspiral
from workflow.Models.Montage import Montage

if __name__ == '__main__':
    show_dag = int(os.environ.get('SHOW_DAG', 0)) == 1
    output_name = str(os.environ.get('OUTPUT_NAME', "dags.json"))
    total_count = int(os.environ.get('TOTAL_COUNT', 1))

    # DAG Type Arguments
    dag_type = os.environ.get('DAG_TYPE', 'Random')
    xml_path = os.environ.get('XML_PATH', "")
    mips_base = int(os.environ.get('MIPS_BASE', 1000))

    dag_size: int = int(os.environ.get('DAG_SIZE', 20))
    dag_communication_min: int = int(os.environ.get('DAG_COMMUNICATION_MIN', 1))
    dag_communication_max: int = int(os.environ.get('DAG_COMMUNICATION_MAX', 20))
    dag_computation_min = int(os.environ.get('DAG_COMPUTATION_MIN', 1000))
    dag_computation_max = int(os.environ.get('DAG_COMPUTATION_MAX', 5000))
    dag_deadline_min = int(os.environ.get('DAG_DEADLINE_MIN', 500))
    dag_deadline_max = int(os.environ.get('DAG_DEADLINE_MAX', 2000))
    dag_memory_min = int(os.environ.get('DAG_MEMORY_MIN', 20))
    dag_memory_max = int(os.environ.get('DAG_MEMORY_MAX', 100))

    for i in range(total_count):
        dag: DAG | None = None
        dag_type_selected = dag_type

        if dag_type_selected == "Pegasus":
            dag = Pegasus.generate_dag(0, xml_path, dag_deadline_min, dag_deadline_max, mips_base)

        else:
            if dag_type_selected == "Random":
                dag_type_selected = random.choice(['Montage', 'CaseWind', 'Genome1000', 'Inspiral', 'CyberShake', 'Epigenomics'])

            elif dag_type_selected == 'Montage':
                dag = Montage.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                           dag_computation_min, dag_computation_max,
                                           dag_deadline_min, dag_deadline_max,
                                           dag_communication_min, dag_communication_max)

            elif dag_type_selected == 'CaseWind':
                dag = CaseWind.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                            dag_computation_min, dag_computation_max,
                                            dag_deadline_min, dag_deadline_max,
                                            dag_communication_min, dag_communication_max)
            elif dag_type_selected == 'Genome1000':
                dag = Genome1000.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                              dag_computation_min, dag_computation_max, dag_deadline_min, dag_deadline_max,
                                              dag_communication_min, dag_communication_max)

            elif dag_type_selected == 'Inspiral':
                dag = Inspiral.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                            dag_computation_min, dag_computation_max, dag_deadline_min, dag_deadline_max,
                                            dag_communication_min, dag_communication_max)

            elif dag_type_selected == 'CyberShake':
                dag = CyberShake.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                              dag_computation_min, dag_computation_max, dag_deadline_min, dag_deadline_max,
                                              dag_communication_min, dag_communication_max)

            elif dag_type_selected == 'Epigenomics':
                dag = Epigenomics.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                               dag_computation_min, dag_computation_max, dag_deadline_min, dag_deadline_max,
                                               dag_communication_min, dag_communication_max)
            else:
                dag = FullTopology.generate_dag(0, dag_size, dag_memory_min, dag_memory_max,
                                               dag_computation_min, dag_computation_max, dag_deadline_min, dag_deadline_max,
                                               dag_communication_min, dag_communication_max)

        DAG.store(dag, "tasks/" + output_name + "/" + f"dag-{i + 1}.json")

        if show_dag:
            dag.show(False)
