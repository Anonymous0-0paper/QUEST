import random

import numpy as np

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class QUEST(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.population_size = 20
        self.max_iterations = 100
        self.mutation_rate = 0.1

        self.gate_fidelity = 0.9999  # 99.99%
        self.coherence_time = 100e-6  # 100 μs

        self.decoherence_rate = 1 / self.coherence_time
        self.error_probability = 1 - self.gate_fidelity

    def run(self):
        population = self.initialize_quantum_population()
        best_assignment = None
        best_objectives = float('inf')
        same_best = 0

        for iteration in range(self.max_iterations):
            if same_best > 5:
                break
            same_best += 1

            new_population = []

            for _ in range(self.population_size):
                parents = self.select_parents(population, k=2)
                child = self.quantum_crossover(parents[0], parents[1])
                child = self.quantum_mutation(child)
                child = self.quantum_interference(child)
                new_population.append(child)

            population = new_population

            for chromosome in population:
                self.measure_quantum_state(chromosome)
                objectives = self.calculate_objectives()
                total_objective = sum(objectives.values())

                if total_objective < best_objectives:
                    best_objectives = total_objective
                    best_assignment = self.assign.copy()
                    same_best = 0

        self.assign = best_assignment
        super().run()

    def initialize_quantum_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for _ in range(len(self.dag.subtasks)):
                probs = np.ones(len(self.network.nodes)) / len(self.network.nodes)
                chromosome.append(probs)
            population.append(chromosome)
        return population

    def measure_quantum_state(self, chromosome):
        for task_id, probs in enumerate(chromosome):
            node_id = np.random.choice(len(self.network.nodes), p=probs)
            self.assign[task_id] = node_id

        super().run()

    def calculate_objectives(self):
        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        data_age = self.calculate_data_age()

        return {
            'energy': energy / 1000,
            'completion_time': completion_time,
            'data_age': data_age * 1000
        }

    def quantum_interference(self, chromosome):
        for i in range(len(chromosome)):
            task_probs = chromosome[i].astype(np.complex128)
            phase = 2 * np.pi * random.random()

            # Apply quantum operations
            task_probs = task_probs * np.exp(1j * phase)
            task_probs = np.abs(task_probs)

            # Safe normalization: prevent division by zero
            sum_probs = np.sum(task_probs)
            if sum_probs > 0:
                task_probs /= sum_probs
            else:
                # If all probabilities are zero, reset to uniform distribution
                task_probs = np.ones_like(task_probs) / len(task_probs)

            chromosome[i] = task_probs
        return chromosome

    def quantum_mutation(self, chromosome):
        if random.random() < self.mutation_rate:
            task_idx = random.randrange(len(chromosome))
            probs = chromosome[task_idx]

            noise = np.random.normal(0, 0.1, size=len(probs))
            probs += noise

            # Ensure non-negative probabilities
            probs = np.abs(probs)

            # Safe normalization
            sum_probs = np.sum(probs)
            if sum_probs > 0:
                probs /= sum_probs
            else:
                # Reset to uniform distribution if all probabilities become zero
                probs = np.ones_like(probs) / len(probs)

            chromosome[task_idx] = probs
        return chromosome

    def quantum_crossover(self, parent1, parent2):
        child = []
        for p1_probs, p2_probs in zip(parent1, parent2):
            alpha = random.random()
            child_probs = alpha * p1_probs + np.sqrt(1 - alpha ** 2) * p2_probs
            child_probs = np.abs(child_probs)
            child_probs /= np.sum(child_probs)
            child.append(child_probs)
        return child

    def fast_non_dominated_sort(self, solutions):
        fronts = [[]]
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

        return fronts[:-1]

    def dominates(self, obj1, obj2):
        better_in_any = False
        for key in ['energy', 'completion_time', 'data_age']:
            if obj1[key] > obj2[key]:
                return False
            elif obj1[key] < obj2[key]:
                better_in_any = True
        return better_in_any

    def calculate_crowding_distance(self, front):
        if len(front) <= 2:
            for solution in front:
                solution[1]['crowding_distance'] = float('inf')
            return

        for solution in front:
            solution[1]['crowding_distance'] = 0

        for objective in ['energy', 'completion_time', 'data_age']:
            front.sort(key=lambda x: x[1][objective])

            obj_range = front[-1][1][objective] - front[0][1][objective]
            if obj_range == 0:
                continue

            front[0][1]['crowding_distance'] = float('inf')
            front[-1][1]['crowding_distance'] = float('inf')

            for i in range(1, len(front) - 1):
                distance = (front[i + 1][1][objective] - front[i - 1][1][objective]) / obj_range
                front[i][1]['crowding_distance'] += distance

    def select_parents(self, population, k=2):
        measured_solutions = []
        for chromosome in population:
            self.measure_quantum_state(chromosome)
            objectives = self.calculate_objectives()
            measured_solutions.append((chromosome, objectives))

        fronts = self.fast_non_dominated_sort(measured_solutions)

        for front in fronts:
            self.calculate_crowding_distance(front)

        selected_parents = []
        for _ in range(k):
            candidates = random.sample(measured_solutions, k=3)
            winner = min(candidates, key=lambda x: (
                x[1].get('rank', float('inf')),
                -x[1].get('crowding_distance', 0)
            ))
            selected_parents.append(winner[0])

        return selected_parents
