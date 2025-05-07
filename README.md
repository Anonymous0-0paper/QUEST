# QUEST: Quantum-inspired Energy-AoI-Aware Task Scheduling in Edge Cloud Continuum

QUEST is a quantum-inspired evolutionary algorithm for optimizing task scheduling in edge-cloud computing environments. It combines quantum computing concepts with evolutionary algorithms to efficiently allocate computational tasks across distributed computing resources while optimizing energy consumption, latency, and Age of Information (AoI).

## Features

- **Quantum-inspired Optimization**: Utilizes quantum computing concepts like superposition, interference, and measurement in a classical implementation
- **Multi-objective Optimization**: Simultaneously optimizes energy consumption, execution time, and data age metrics
- **Dynamic Frequency Scaling**: Employs TODVFS (Task-Oriented Dynamic Voltage and Frequency Scaling) for further energy optimization
- **Convergence Analysis**: Provides comprehensive tools for convergence analysis and algorithm performance evaluation
<!-- - **Sensitivity Analysis**: Includes tools for parameter optimization through sensitivity analysis-->
- **Comparative Evaluation**: Supports comparison with multiple baseline algorithms

## Algorithms

The framework implements several scheduling algorithms:

- **QUEST**: Quantum-inspired Evolutionary Algorithm for Optimization
- **NSGA3**: Non-dominated Sorting Genetic Algorithm III
- **MOPSO**: Multi-Objective Particle Swarm Optimization
- **MQGA**: Multi-objective Quantum Genetic Algorithm
- **Greedy**: Simple greedy allocation approach
- **Fuzzy**: Fuzzy logic-based allocation
- **Random**: Random allocation (baseline)
- **RL**: ID3CO-based approach

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/QUEST.git
cd QUEST
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

You can run experiments using the main script:

```bash
python experiment/main.py 
```

### Generating Plots

The framework provides a specialized script for generating performance plots:

```bash
python generate_plots.py
```

This will generate multiple plots in the `plots/` directory:

1. **Convergence Plots**: Shows how the algorithm converges over iterations
2. **Sensitivity Analysis Plots**: Shows how different parameter values affect algorithm performance

## Configuration

Experiments are configured using INI files in the `experiment/` directory. Key parameters include:

- `algorithms`: List of algorithms to compare
- `iteration`: Number of iterations to run
- `regions`: Network topology regions (e.g., "Lyon, Luxembourg, Toulouse")
- Nodes and connections specifications
- Workload specifications

## Project Structure

- `/algorithms`: Implementation of all scheduling algorithms
- `/model`: Data models including DAG and SubTask models
- `/network`: Network topology and connection models
- `/experiment`: Experiment execution and analysis code
- `/plots`: Generated visualization plots
- `/tasks`: Task DAG definitions
- `/csv_results`: Raw results in CSV format

## Visualization

The framework includes comprehensive visualization capabilities for analyzing algorithm performance and optimization parameters. The visualizations are designed to provide insights into the convergence behavior, sensitivity to parameters, and comparative performance of different algorithms.

### Convergence Analysis

Convergence plots show how the objective metrics improve over iterations:

1. **Individual Run Plots**: For each algorithm and workload combination, individual run plots show the improvement rates and normalized fitness values over iterations for each run.

2. **Average Convergence Plots**: These plots show the average improvement rate and fitness values across all runs, with standard deviation bands to illustrate the variance in performance.

Key metrics visualized:
- **Improvement Rate**: The percentage improvement in objective values between consecutive iterations
- **Normalized Fitness**: The objective values normalized to a [0,1] scale for easier comparison

### Sensitivity Analysis

Sensitivity analysis plots help identify optimal parameter settings:

1. **Parameter-specific Plots**: Individual plots for each parameter (mutation_rate, population_size, elite_size) showing how different values affect the algorithm's performance.

2. **Comparative Plot**: A normalized plot that combines all parameters on the same scale to directly compare their relative impact.

Key features of sensitivity plots:
- **Best Value Highlighting**: The optimal value for each parameter is highlighted with a star marker
- **Gradient Shading**: Visual emphasis on the relationship between parameter values and performance
- **Annotations**: Descriptive annotations explaining the significance of key points

### MARL Information Sharing Analysis

The framework also includes capabilities for analyzing Multi-Agent Reinforcement Learning (MARL) information sharing strategies:

1. **Communication Overhead**: Comparison between partial and full information sharing approaches in terms of bytes transferred during training.

2. **Reduction Ratio**: How the information reduction ratio improves over time with partial sharing.

3. **Agent Scaling**: Analysis of how communication overhead scales with an increasing number of agents.

4. **Performance Comparison**: Comparative analysis of key performance metrics (energy, latency, throughput, convergence time).

5. **Deployment Scenarios**: Analysis of communication overhead across different regional deployment configurations.

These visualizations clearly demonstrate that partial information sharing in MARL is often more efficient than full information sharing, providing comparable or better performance while significantly reducing communication overhead.

## Project Structure

- `/algorithms`: Implementation of all scheduling algorithms
- `/model`: Data models including DAG and SubTask models
- `/network`: Network topology and connection models
- `/experiment`: Experiment execution and analysis code
- `/plots`: Generated visualization plots
- `/tasks`: Task DAG definitions
- `/csv_results`: Raw results in CSV format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation
The paper is under review!!!
<!---If you use QUEST in your research, please cite [Note: ]:-->

```
@article{quest2025,
  title={QUEST: Quantum-inspired Energy-AoI-Aware Task Scheduling in Edge Cloud Continuum},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

