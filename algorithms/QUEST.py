# import random
#
# import numpy as np
#
# from algorithms.Algorithm import Algorithm
# from model.DAGModel import DAGModel
# from network.Network import Network
#
#
# class QUEST(Algorithm):
#     def __init__(self, network: Network, dag: DAGModel):
#         super().__init__(network, dag)
#         self.population_size = 20
#         self.max_iterations = 100
#         self.mutation_rate = 0.1
#
#         self.gate_fidelity = 0.9999  # 99.99%
#         self.coherence_time = 100e-6  # 100 μs
#
#         self.decoherence_rate = 1 / self.coherence_time
#         self.error_probability = 1 - self.gate_fidelity
#
#     def run(self):
#         population = self.initialize_quantum_population()
#         best_assignment = None
#         best_objectives = float('inf')
#         same_best = 0
#
#         for iteration in range(self.max_iterations):
#             if same_best > 5:
#                 break
#             same_best += 1
#
#             new_population = []
#
#             for _ in range(self.population_size):
#                 parents = self.select_parents(population, k=2)
#                 child = self.quantum_crossover(parents[0], parents[1])
#                 child = self.quantum_mutation(child)
#                 child = self.quantum_interference(child)
#                 new_population.append(child)
#
#             population = new_population
#
#             for chromosome in population:
#                 self.measure_quantum_state(chromosome)
#                 objectives = self.calculate_objectives()
#                 total_objective = sum(objectives.values())
#
#                 if total_objective < best_objectives:
#                     best_objectives = total_objective
#                     best_assignment = self.assign.copy()
#                     same_best = 0
#
#         self.assign = best_assignment
#         super().run()
#
#     def initialize_quantum_population(self):
#         population = []
#         for _ in range(self.population_size):
#             chromosome = []
#             for _ in range(len(self.dag.subtasks)):
#                 probs = np.ones(len(self.network.nodes)) / len(self.network.nodes)
#                 chromosome.append(probs)
#             population.append(chromosome)
#         return population
#
#     def measure_quantum_state(self, chromosome):
#         for task_id, probs in enumerate(chromosome):
#             node_id = np.random.choice(len(self.network.nodes), p=probs)
#             self.assign[task_id] = node_id
#
#         super().run()
#
#     def calculate_objectives(self):
#         energy = self.calculate_energy()
#         completion_time = self.calculate_completion_time()
#         data_age = self.calculate_data_age()
#
#         return {
#             'energy': energy / 1000,
#             'completion_time': completion_time,
#             'data_age': data_age * 1000
#         }
#
#     def quantum_interference(self, chromosome):
#         for i in range(len(chromosome)):
#             task_probs = chromosome[i].astype(np.complex128)
#             phase = 2 * np.pi * random.random()
#
#             # Apply quantum operations
#             task_probs = task_probs * np.exp(1j * phase)
#             task_probs = np.abs(task_probs)
#
#             # Safe normalization: prevent division by zero
#             sum_probs = np.sum(task_probs)
#             if sum_probs > 0:
#                 task_probs /= sum_probs
#             else:
#                 # If all probabilities are zero, reset to uniform distribution
#                 task_probs = np.ones_like(task_probs) / len(task_probs)
#
#             chromosome[i] = task_probs
#         return chromosome
#
#     def quantum_mutation(self, chromosome):
#         if random.random() < self.mutation_rate:
#             task_idx = random.randrange(len(chromosome))
#             probs = chromosome[task_idx]
#
#             noise = np.random.normal(0, 0.1, size=len(probs))
#             probs += noise
#
#             # Ensure non-negative probabilities
#             probs = np.abs(probs)
#
#             # Safe normalization
#             sum_probs = np.sum(probs)
#             if sum_probs > 0:
#                 probs /= sum_probs
#             else:
#                 # Reset to uniform distribution if all probabilities become zero
#                 probs = np.ones_like(probs) / len(probs)
#
#             chromosome[task_idx] = probs
#         return chromosome
#
#     def quantum_crossover(self, parent1, parent2):
#         child = []
#         for p1_probs, p2_probs in zip(parent1, parent2):
#             alpha = random.random()
#             child_probs = alpha * p1_probs + np.sqrt(1 - alpha ** 2) * p2_probs
#             child_probs = np.abs(child_probs)
#             child_probs /= np.sum(child_probs)
#             child.append(child_probs)
#         return child
#
#     def fast_non_dominated_sort(self, solutions):
#         fronts = [[]]
#         for p in solutions:
#             p[1]['domination_count'] = 0
#             p[1]['dominated_solutions'] = []
#
#             for q in solutions:
#                 if self.dominates(p[1], q[1]):
#                     p[1]['dominated_solutions'].append(q)
#                 elif self.dominates(q[1], p[1]):
#                     p[1]['domination_count'] += 1
#
#             if p[1]['domination_count'] == 0:
#                 p[1]['rank'] = 0
#                 fronts[0].append(p)
#
#         i = 0
#         while fronts[i]:
#             next_front = []
#             for p in fronts[i]:
#                 for q in p[1]['dominated_solutions']:
#                     q[1]['domination_count'] -= 1
#                     if q[1]['domination_count'] == 0:
#                         q[1]['rank'] = i + 1
#                         next_front.append(q)
#             i += 1
#             fronts.append(next_front)
#
#         return fronts[:-1]
#
#     def dominates(self, obj1, obj2):
#         better_in_any = False
#         for key in ['energy', 'completion_time', 'data_age']:
#             if obj1[key] > obj2[key]:
#                 return False
#             elif obj1[key] < obj2[key]:
#                 better_in_any = True
#         return better_in_any
#
#     def calculate_crowding_distance(self, front):
#         if len(front) <= 2:
#             for solution in front:
#                 solution[1]['crowding_distance'] = float('inf')
#             return
#
#         for solution in front:
#             solution[1]['crowding_distance'] = 0
#
#         for objective in ['energy', 'completion_time', 'data_age']:
#             front.sort(key=lambda x: x[1][objective])
#
#             obj_range = front[-1][1][objective] - front[0][1][objective]
#             if obj_range == 0:
#                 continue
#
#             front[0][1]['crowding_distance'] = float('inf')
#             front[-1][1]['crowding_distance'] = float('inf')
#
#             for i in range(1, len(front) - 1):
#                 distance = (front[i + 1][1][objective] - front[i - 1][1][objective]) / obj_range
#                 front[i][1]['crowding_distance'] += distance
#
#     def select_parents(self, population, k=2):
#         measured_solutions = []
#         for chromosome in population:
#             self.measure_quantum_state(chromosome)
#             objectives = self.calculate_objectives()
#             measured_solutions.append((chromosome, objectives))
#
#         fronts = self.fast_non_dominated_sort(measured_solutions)
#
#         for front in fronts:
#             self.calculate_crowding_distance(front)
#
#         selected_parents = []
#         for _ in range(k):
#             candidates = random.sample(measured_solutions, k=3)
#             winner = min(candidates, key=lambda x: (
#                 x[1].get('rank', float('inf')),
#                 -x[1].get('crowding_distance', 0)
#             ))
#             selected_parents.append(winner[0])
#
#         return selected_parents
#

import random
import logging
import copy
from typing import List, Tuple, Dict, Any

import numpy as np

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class QUEST(Algorithm):
    """
    QUEST: Quantum-inspired Evolutionary Algorithm for Optimization.

    This algorithm utilizes quantum operations such as quantum crossover,
    mutation, and interference to evolve a population of solutions.
    """

    def __init__(self, network: Network, dag: DAGModel):
        """
        Initializes the QUEST algorithm with the given network and DAG model.

        Args:
            network (Network): The network model.
            dag (DAGModel): The DAG model representing subtasks.
        """
        super().__init__(network, dag)
        self.population_size: int = 50
        self.max_iterations: int = 200
        self.mutation_rate: float = 0.1
        self.elite_size: int = 2  # Number of elite individuals to preserve

        # Quantum parameters
        self.gate_fidelity: float = 0.9999  # 99.99%
        self.coherence_time: float = 100e-6  # 100 μs
        self.decoherence_rate: float = 1 / self.coherence_time
        self.error_probability: float = 1 - self.gate_fidelity

        # Initialize assignment as a dictionary
        self.assign: Dict[int, int] = {subtask.id: 0 for subtask in self.dag.subtasks}

    def run(self):
        """
        Executes the QUEST algorithm to find the optimal assignment of subtasks to network nodes.
        """
        population = self.initialize_quantum_population()
        best_assignment = None
        best_objectives = float('inf')
        same_best_counter = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Iteration {iteration}/{self.max_iterations}")

            # Evaluate current population
            evaluated_population = self.evaluate_population(population)

            # Identify elites
            elites = self.get_elites(evaluated_population, self.elite_size)
            logger.debug(f"Elites selected: {self.elite_size}")

            # Find the best solution in the current population
            current_best = min(evaluated_population, key=lambda x: x[1]['total_objective'])
            current_best_objective = current_best[1]['total_objective']

            if current_best_objective < best_objectives:
                best_objectives = current_best_objective
                best_assignment = current_best[2].copy()  # Assignment corresponding to the best chromosome
                same_best_counter = 0
                logger.info(f"New best objective: {best_objectives}")
            else:
                same_best_counter += 1
                logger.debug(f"No improvement in this iteration. Counter: {same_best_counter}")
                self.adapt_mutation_rate(same_best_counter)

            # Check for convergence
            if same_best_counter > 10:
                logger.info("Convergence criteria met. Stopping algorithm.")
                break

            # Generate new population
            new_population = elites.copy()  # Preserve elites

            while len(new_population) < self.population_size:
                parents = self.select_parents(evaluated_population, k=2)
                child = self.quantum_crossover(parents[0], parents[1])
                child = self.quantum_mutation(child)
                child = self.quantum_interference(child)
                new_population.append(child)

            population = new_population

        if best_assignment is not None:
            self.assign = best_assignment
            logger.info("Best assignment found and set.")
        else:
            logger.warning("No improvement found during QUEST run. Using last population's assignment.")

        logger.info("QUEST algorithm completed.")
        super().run()


    def initialize_quantum_population(self) -> List[List[np.ndarray]]:
        """
        Initializes the quantum population with uniform probability distributions.

        Returns:
            List[List[np.ndarray]]: The initial population.
        """
        population = []
        num_tasks = len(self.dag.subtasks)
        num_nodes = len(self.network.nodes)

        uniform_prob = np.ones(num_nodes) / num_nodes
        for _ in range(self.population_size):
            chromosome = [uniform_prob.copy() for _ in range(num_tasks)]
            population.append(chromosome)

        logger.debug("Quantum population initialized.")
        return population

    # def evaluate_population(
    #         self, population: List[List[np.ndarray]]
    # ) -> List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]:
    #     """
    #     Measures and evaluates each chromosome in the population.
    #
    #     Args:
    #         population (List[List[np.ndarray]]): The current population.
    #
    #     Returns:
    #         List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]: The evaluated population with objectives and assignments.
    #     """
    #     evaluated = []
    #     for idx, chromosome in enumerate(population):
    #         # Create deep copies of network and dag for independent evaluation
    #         temp_network = copy.deepcopy(self.network)
    #         temp_dag = copy.deepcopy(self.dag)
    #
    #         # Instantiate a temporary Algorithm with the copied network and dag
    #         temp_algorithm = Algorithm(temp_network, temp_dag)
    #
    #         # Set the temporary assignment based on the chromosome
    #         temp_algorithm.assign = self.chromosome_to_assignment(chromosome)
    #
    #         # Run the scheduling algorithm on the temporary instance
    #         temp_algorithm.run()
    #
    #         # Calculate objectives using the temporary Algorithm's methods
    #         energy = temp_algorithm.calculate_energy()
    #         completion_time = temp_algorithm.calculate_completion_time()
    #         data_age = temp_algorithm.calculate_data_age()
    #
    #         # Aggregate objectives
    #         objectives = {
    #             'energy': energy / 1000,            # Convert to appropriate units
    #             'completion_time': completion_time,
    #             'data_age': data_age * 1000          # Convert to appropriate units
    #         }
    #         total_objective = sum(objectives.values())
    #
    #         # Append the evaluated chromosome with its objectives and assignment
    #         evaluated.append((chromosome, {**objectives, 'total_objective': total_objective}, temp_algorithm.assign))
    #
    #         logger.debug(f"Chromosome {idx} evaluated with total objective {total_objective}.")
    #
    #     logger.debug("Population evaluated.")
    #     return evaluated

    # In evaluate_population, remove total_objective
    def evaluate_population(
            self, population: List[List[np.ndarray]]
    ) -> List[Tuple[List[np.ndarray], Dict[str, float], Dict[int, int]]]:
        evaluated = []
        for idx, chromosome in enumerate(population):
            # Deep copies for independent evaluation
            temp_network = copy.deepcopy(self.network)
            temp_dag = copy.deepcopy(self.dag)

            # Temporary Algorithm instance
            temp_algorithm = Algorithm(temp_network, temp_dag)

            # Set temporary assignment
            temp_algorithm.assign = self.chromosome_to_assignment(chromosome)

            # Run scheduling
            temp_algorithm.run()

            # Calculate objectives
            energy = temp_algorithm.calculate_energy()
            completion_time = temp_algorithm.calculate_completion_time()
            data_age = temp_algorithm.calculate_data_age()

            # Aggregate objectives with weights
            weights = {
                'energy': 0.5,
                'completion_time': 0.3,
                'data_age': 0.2
            }
            objectives = {
                'energy': (energy / 1000) * weights['energy'],
                'completion_time': completion_time * weights['completion_time'],
                'data_age': (data_age * 1000) * weights['data_age']
            }

            # Calculate total_objective as the sum of weighted objectives
            objectives['total_objective'] = sum(objectives.values())

            # Append evaluated chromosome
            evaluated.append((chromosome, objectives, temp_algorithm.assign))

            logger.debug(f"Chromosome {idx} evaluated with objectives: {objectives}.")

        logger.debug("Population evaluated.")
        return evaluated


    def chromosome_to_assignment(self, chromosome: List[np.ndarray]) -> Dict[int, int]:
        """
        Converts a chromosome to an assignment dictionary mapping subtask IDs to node IDs.

        Args:
            chromosome (List[np.ndarray]): The quantum chromosome.

        Returns:
            Dict[int, int]: The assignment mapping subtask IDs to node IDs.
        """
        assignment = {}
        for subtask, probs in zip(self.dag.subtasks, chromosome):
            node_id = np.random.choice(len(self.network.nodes), p=probs)
            assignment[subtask.id] = node_id
        return assignment

    # def get_elites(
    #         self, evaluated_population: List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]], elite_size: int
    # ) -> List[List[np.ndarray]]:
    #     """
    #     Selects the top elite chromosomes from the evaluated population.
    #
    #     Args:
    #         evaluated_population (List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]): The evaluated population.
    #         elite_size (int): Number of elite individuals to select.
    #
    #     Returns:
    #         List[List[np.ndarray]]: The elite chromosomes.
    #     """
    #     sorted_population = sorted(evaluated_population, key=lambda x: x[1]['total_objective'])
    #     elites = [chromosome for chromosome, _, _ in sorted_population[:elite_size]]
    #     return elites
    def get_elites(
            self, evaluated_population: List[Tuple[List[np.ndarray], Dict[str, float], Dict[int, int]]],
            elite_size: int
    ) -> List[List[np.ndarray]]:
        """
        Selects the top elite chromosomes based on Pareto dominance.

        Args:
            evaluated_population: The evaluated population with separate objectives.
            elite_size: Number of elite individuals to select.

        Returns:
            The elite chromosomes.
        """
        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(evaluated_population)

        # Select elites from the first front
        elites = []
        for front in fronts:
            if len(elites) + len(front) > elite_size:
                # If adding the entire front exceeds elite_size, sort by crowding distance
                front_sorted = sorted(front, key=lambda x: x[1].get('crowding_distance', 0), reverse=True)
                elites.extend([chromosome for chromosome, _, _ in front_sorted[:elite_size - len(elites)]])
                break
            elites.extend([chromosome for chromosome, _, _ in front])
        return elites

    def measure_quantum_state(self, chromosome: List[np.ndarray]) -> Dict[int, int]:
        """
        Measures the quantum state of a chromosome to determine task assignments.

        Args:
            chromosome (List[np.ndarray]): The quantum chromosome.

        Returns:
            Dict[int, int]: The assignment mapping subtask IDs to node IDs.
        """
        assign_copy = {}
        for subtask, probs in zip(self.dag.subtasks, chromosome):
            node_id = np.random.choice(len(self.network.nodes), p=probs)
            assign_copy[subtask.id] = node_id
        logger.debug("Quantum state measured.")
        return assign_copy

    # def calculate_objectives(self, assignment: Dict[int, int]) -> Dict[str, float]:
    #     """
    #     Calculates the objective metrics for the given assignment.
    #
    #     Args:
    #         assignment (Dict[int, int]): The assignment mapping subtask IDs to node IDs.
    #
    #     Returns:
    #         Dict[str, float]: The calculated objectives.
    #     """
    #     # Temporarily set self.assign to the given assignment
    #     original_assign = self.assign.copy()
    #     self.assign = assignment.copy()
    #
    #     energy = self.calculate_energy()
    #     completion_time = self.calculate_completion_time()
    #     data_age = self.calculate_data_age()
    #
    #     # Restore original assignment
    #     self.assign = original_assign
    #
    #     objectives = {
    #         'energy': energy / 1000,            # Convert to appropriate units
    #         'completion_time': completion_time,
    #         'data_age': data_age * 1000          # Convert to appropriate units
    #     }
    #     logger.debug(f"Objectives calculated: {objectives}")
    #     return objectives

    def calculate_objectives(self, assignment: Dict[int, int]) -> Dict[str, float]:
        """
        Calculates the objective metrics for the given assignment with weights.

        Args:
            assignment: The assignment mapping subtask IDs to node IDs.

        Returns:
            The weighted objectives.
        """
        # Define weights
        weights = {
            'energy': 0.5,
            'completion_time': 0.3,
            'data_age': 0.2
        }

        # Temporarily set self.assign to the given assignment
        original_assign = self.assign.copy()
        self.assign = assignment.copy()

        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        data_age = self.calculate_data_age()

        # Restore original assignment
        self.assign = original_assign

        # Apply weights
        objectives = {
            'energy': (energy / 1000) * weights['energy'],
            'completion_time': completion_time * weights['completion_time'],
            'data_age': (data_age * 1000) * weights['data_age']
        }

        logger.debug(f"Weighted objectives calculated: {objectives}")
        return objectives


    def quantum_interference(self, chromosome: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies quantum interference to a chromosome.

        Args:
            chromosome (List[np.ndarray]): The quantum chromosome.

        Returns:
            List[np.ndarray]: The updated chromosome after interference.
        """
        for i in range(len(chromosome)):
            task_probs = chromosome[i].astype(np.complex128)
            phase = 2 * np.pi * random.random()

            # Apply quantum phase rotation
            task_probs *= np.exp(1j * phase)
            task_probs = np.abs(task_probs)

            # Normalize probabilities safely
            sum_probs = task_probs.sum()
            if sum_probs > 0:
                task_probs /= sum_probs
            else:
                task_probs = np.ones_like(task_probs) / len(task_probs)

            chromosome[i] = task_probs.real  # Ensure probabilities are real numbers

        logger.debug("Quantum interference applied.")
        return chromosome

    def adapt_mutation_rate(self, same_best_counter: int):
        """
        Adjusts the mutation rate based on the convergence criteria.

        Args:
            same_best_counter: Counter tracking iterations without improvement.
        """
        if same_best_counter > 5:
            self.mutation_rate = min(self.mutation_rate * 1.5, 1.0)  # Increase mutation rate
            logger.info(f"Mutation rate increased to {self.mutation_rate}.")
        elif same_best_counter < 3 and self.mutation_rate > 0.01:
            self.mutation_rate *= 0.9  # Decrease mutation rate
            logger.info(f"Mutation rate decreased to {self.mutation_rate}.")

    # def quantum_mutation(self, chromosome: List[np.ndarray]) -> List[np.ndarray]:
    #     """
    #     Applies quantum mutation to a chromosome.
    #
    #     Args:
    #         chromosome (List[np.ndarray]): The quantum chromosome.
    #
    #     Returns:
    #         List[np.ndarray]: The mutated chromosome.
    #     """
    #     if random.random() < self.mutation_rate:
    #         subtask = random.choice(self.dag.subtasks)
    #         task_id = subtask.id
    #
    #         # Find the index of the subtask in the DAG
    #         try:
    #             task_index = next(i for i, st in enumerate(self.dag.subtasks) if st.id == task_id)
    #         except StopIteration:
    #             logger.error(f"Subtask ID {task_id} not found in DAG.")
    #             return chromosome
    #
    #         probs = chromosome[task_index].copy()
    #
    #         # Add Gaussian noise
    #         noise = np.random.normal(0, 0.1, size=probs.shape)
    #         probs += noise
    #
    #         # Ensure non-negative probabilities
    #         probs = np.clip(probs, a_min=1e-10, a_max=None)
    #
    #         # Normalize probabilities
    #         sum_probs = probs.sum()
    #         if sum_probs > 0:
    #             probs /= sum_probs
    #         else:
    #             probs = np.ones_like(probs) / len(probs)
    #
    #         chromosome[task_index] = probs
    #         logger.debug(f"Quantum mutation applied to subtask ID {task_id} (index {task_index}).")
    #
    #     return chromosome

    def quantum_mutation(self, chromosome: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies controlled Gaussian mutation to a chromosome.

        Args:
            chromosome: The quantum chromosome.

        Returns:
            The mutated chromosome.
        """
        for task_index in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                mutation_strength = 0.1  # Controlled perturbation
                noise = np.random.normal(0, mutation_strength, size=chromosome[task_index].shape)
                mutated_probs = chromosome[task_index] + noise
                # Ensure non-negative probabilities
                mutated_probs = np.clip(mutated_probs, a_min=1e-10, a_max=None)
                # Normalize
                sum_probs = mutated_probs.sum()
                if sum_probs > 0:
                    mutated_probs /= sum_probs
                else:
                    mutated_probs = np.ones_like(mutated_probs) / len(mutated_probs)
                chromosome[task_index] = mutated_probs
                logger.debug(f"Quantum mutation applied to task index {task_index}.")
        return chromosome

    # def quantum_crossover(
    #         self, parent1: List[np.ndarray], parent2: List[np.ndarray]
    # ) -> List[np.ndarray]:
    #     """
    #     Performs quantum crossover between two parent chromosomes to produce a child.
    #
    #     Args:
    #         parent1 (List[np.ndarray]): The first parent chromosome.
    #         parent2 (List[np.ndarray]): The second parent chromosome.
    #
    #     Returns:
    #         List[np.ndarray]: The child chromosome.
    #     """
    #     child = []
    #     for p1_probs, p2_probs in zip(parent1, parent2):
    #         alpha = random.random()
    #         child_probs = alpha * p1_probs + np.sqrt(1 - alpha ** 2) * p2_probs
    #         child_probs = np.abs(child_probs)
    #
    #         # Normalize to form a valid probability distribution
    #         sum_probs = child_probs.sum()
    #         if sum_probs > 0:
    #             child_probs /= sum_probs
    #         else:
    #             child_probs = np.ones_like(child_probs) / len(child_probs)
    #
    #         child.append(child_probs)
    #     logger.debug("Quantum crossover performed.")
    #     return child

    def quantum_crossover(
            self, parent1: List[np.ndarray], parent2: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Performs uniform quantum crossover between two parent chromosomes to produce a child.

        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.

        Returns:
            The child chromosome.
        """
        child = []
        for p1_probs, p2_probs in zip(parent1, parent2):
            mask = np.random.rand(len(p1_probs)) < 0.5
            child_probs = np.where(mask, p1_probs, p2_probs)
            # Normalize to form a valid probability distribution
            sum_probs = child_probs.sum()
            if sum_probs > 0:
                child_probs /= sum_probs
            else:
                child_probs = np.ones_like(child_probs) / len(child_probs)
            child.append(child_probs)
        logger.debug("Quantum uniform crossover performed.")
        return child

    def fast_non_dominated_sort(
            self, solutions: List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]
    ) -> List[List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]]:
        """
        Performs fast non-dominated sorting on the given solutions.

        Args:
            solutions (List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]): The solutions to sort.

        Returns:
            List[List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]]: The sorted fronts.
        """
        fronts: List[List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]] = [[]]
        for p in solutions:
            p[1]['domination_count'] = 0
            p[1]['dominated_solutions'] = []

            for q in solutions:
                if self.dominates(p[1], q[1]):
                    p[1]['dominated_solutions'].append(q)
                elif self.dominates(q[1], p[1]):
                    p[1]['domination_count'] += 1

            if p[1]['domination_count'] == 0:
                p[1]['rank'] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p[1]['dominated_solutions']:
                    q[1]['domination_count'] -= 1
                    if q[1]['domination_count'] == 0:
                        q[1]['rank'] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        logger.debug("Fast non-dominated sort completed.")
        return fronts[:-1]

    def dominates(
            self, obj1: Dict[str, float], obj2: Dict[str, float]
    ) -> bool:
        """
        Determines if one objective set dominates another.

        Args:
            obj1 (Dict[str, float]): The first set of objectives.
            obj2 (Dict[str, float]): The second set of objectives.

        Returns:
            bool: True if obj1 dominates obj2, False otherwise.
        """
        better_in_any = False
        for key in ['energy', 'completion_time', 'data_age']:
            if obj1[key] > obj2[key]:
                return False
            elif obj1[key] < obj2[key]:
                better_in_any = True
        return better_in_any

    def calculate_crowding_distance(
            self, front: List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]
    ):
        """
        Calculates the crowding distance for each solution in a front.

        Args:
            front (List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]): The front of solutions.
        """
        if len(front) <= 2:
            for solution in front:
                solution[1]['crowding_distance'] = float('inf')
            return

        for solution in front:
            solution[1]['crowding_distance'] = 0.0

        for objective in ['energy', 'completion_time', 'data_age']:
            front.sort(key=lambda x: x[1][objective])
            min_obj = front[0][1][objective]
            max_obj = front[-1][1][objective]
            obj_range = max_obj - min_obj

            if obj_range == 0:
                continue

            front[0][1]['crowding_distance'] = float('inf')
            front[-1][1]['crowding_distance'] = float('inf')

            for i in range(1, len(front) - 1):
                distance = (front[i + 1][1][objective] - front[i - 1][1][objective]) / obj_range
                front[i][1]['crowding_distance'] += distance

        logger.debug("Crowding distance calculated.")

    def select_parents(
            self, evaluated_population: List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]], k: int = 2
    ) -> List[List[np.ndarray]]:
        """
        Selects parent chromosomes for crossover using tournament selection.

        Args:
            evaluated_population (List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]): The evaluated population.
            k (int): Number of parents to select.

        Returns:
            List[List[np.ndarray]]: The selected parent chromosomes.
        """
        fronts = self.fast_non_dominated_sort(evaluated_population)

        # Assign ranks and calculate crowding distances
        for front in fronts:
            self.calculate_crowding_distance(front)

        selected_parents = []
        while len(selected_parents) < k:
            tournament = random.sample(evaluated_population, 3)
            winner = self.tournament_selection(tournament)
            selected_parents.append(winner[0])

        logger.debug(f"Parents selected: {k}")
        return selected_parents

    def tournament_selection(
            self, tournament: List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]:
        """
        Performs tournament selection among a group of solutions.

        Args:
            tournament (List[Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]]): The tournament participants.

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any], Dict[int, int]]: The winning solution.
        """
        # Sort by rank and crowding distance
        tournament_sorted = sorted(
            tournament,
            key=lambda x: (x[1].get('rank', float('inf')), -x[1].get('crowding_distance', 0))
        )
        winner = tournament_sorted[0]
        return winner
