import random
import numpy as np
from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class MQGA(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.population_size = 20
        self.max_iterations = 100
        self.qubit_num = len(self.dag.subtasks)
        self.vm_num = len(self.network.nodes)
        self.min_rotation_angle = 0.01 * np.pi
        self.max_rotation_angle = 0.05 * np.pi
        self.epsilon = 1e-10

    def run(self):
        population = self.initialize_quantum_population()
        best_solution = None
        best_objectives = float('inf')  # Only compare energy and completion time
        generation_without_improvement = 0

        while generation_without_improvement < self.max_iterations:
            classical_solutions = self.quantum_measure(population)
            evaluated_solutions = self.evaluate_solutions(classical_solutions)
            fronts = self.fast_non_dominated_sort(evaluated_solutions)

            for front in fronts:
                self.calculate_crowding_distance(front)

            current_best = self.select_best_solution(evaluated_solutions)
            current_objectives = current_best[1]['energy'] + current_best[1]['completion_time']

            if current_objectives < best_objectives:
                best_solution = current_best[0].copy()
                best_objectives = current_objectives
                generation_without_improvement = 0
            else:
                generation_without_improvement += 1

            population = self.quantum_rotation(population, best_solution, generation_without_improvement)
            population = self.quantum_mutation(population)

        self.assign = best_solution
        super().run()

    def initialize_quantum_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for _ in range(self.qubit_num):
                qubits_for_vm = []
                angle = np.random.uniform(0, 2 * np.pi)
                base_alpha = np.cos(angle)
                base_beta = np.sin(angle)

                for _ in range(self.vm_num):
                    noise = np.random.normal(0, 0.01)
                    alpha = np.clip(base_alpha + noise, -1, 1)
                    beta = np.sqrt(1 - alpha ** 2) * (1 if base_beta >= 0 else -1)
                    qubits_for_vm.append([alpha, beta])
                chromosome.append(qubits_for_vm)
            population.append(chromosome)
        return population

    def quantum_measure(self, population):
        classical_solutions = []

        for chromosome in population:
            solution = {}
            for task_id, qubits_for_vm in enumerate(chromosome):
                probabilities = [max(0, alpha ** 2) for alpha, _ in qubits_for_vm]
                total_prob = sum(probabilities)

                if total_prob < self.epsilon:
                    probabilities = [1 / self.vm_num] * self.vm_num
                else:
                    probabilities = [p / total_prob for p in probabilities]

                solution[task_id] = np.random.choice(range(self.vm_num), p=probabilities)
            classical_solutions.append(solution)

        return classical_solutions

    def evaluate_solutions(self, classical_solutions):
        evaluated_solutions = []
        for solution in classical_solutions:
            self.assign = solution
            super().run()
            objectives = {
                'energy': self.calculate_energy(),
                'completion_time': self.calculate_completion_time(),
                'dominated_solutions': [],
                'domination_count': 0,
                'crowding_distance': 0
            }
            evaluated_solutions.append((solution, objectives))
        return evaluated_solutions

    def quantum_rotation(self, population, best_solution, generation):
        adaptive_angle = self.min_rotation_angle + \
                         (self.max_rotation_angle - self.min_rotation_angle) * \
                         np.exp(-generation / self.max_iterations)

        new_population = []
        for chromosome in population:
            new_chromosome = []
            for task_id, qubits_for_vm in enumerate(chromosome):
                new_qubits = []
                best_vm = best_solution[task_id]

                for vm_id, qubit in enumerate(qubits_for_vm):
                    alpha, beta = qubit
                    rotation_direction = 1 if vm_id == best_vm else -1
                    angle = rotation_direction * adaptive_angle

                    new_alpha = alpha * np.cos(angle) - beta * np.sin(angle)
                    new_beta = alpha * np.sin(angle) + beta * np.cos(angle)

                    magnitude = np.sqrt(new_alpha ** 2 + new_beta ** 2)
                    if magnitude > self.epsilon:
                        new_alpha /= magnitude
                        new_beta /= magnitude
                    else:
                        new_alpha = 1 / np.sqrt(2)
                        new_beta = 1 / np.sqrt(2)

                    new_qubits.append([new_alpha, new_beta])
                new_chromosome.append(new_qubits)
            new_population.append(new_chromosome)
        return new_population

    def quantum_mutation(self, population):
        mutation_rate = 0.1
        for chromosome in population:
            if random.random() < mutation_rate:
                task_id = random.randrange(self.qubit_num)
                vm_id = random.randrange(self.vm_num)
                angle = random.uniform(0, 2 * np.pi)

                chromosome[task_id][vm_id][0] = np.cos(angle)
                chromosome[task_id][vm_id][1] = np.sin(angle)

                for other_vm in range(self.vm_num):
                    if other_vm != vm_id:
                        chromosome[task_id][other_vm][0] = np.sin(angle) / np.sqrt(self.vm_num - 1)
                        chromosome[task_id][other_vm][1] = np.cos(angle) / np.sqrt(self.vm_num - 1)
        return population

    def fast_non_dominated_sort(self, solutions):
        fronts = [[]]
        dominated_set = {i: set() for i in range(len(solutions))}
        domination_count = [0] * len(solutions)

        for i, (_, objectives_i) in enumerate(solutions):
            for j, (_, objectives_j) in enumerate(solutions):
                if i != j:
                    if self.dominates(objectives_i, objectives_j):
                        dominated_set[i].add(j)
                    elif self.dominates(objectives_j, objectives_i):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                objectives_i['rank'] = 0
                fronts[0].append(solutions[i])

        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for solution_idx, solution in enumerate(fronts[front_idx]):
                for dominated_idx in dominated_set[solutions.index(solution)]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        solutions[dominated_idx][1]['rank'] = front_idx + 1
                        next_front.append(solutions[dominated_idx])
            front_idx += 1
            fronts.append(next_front)

        return [front for front in fronts if front]

    def dominates(self, obj1, obj2):
        better_in_any = False
        for key in ['energy', 'completion_time']:
            if obj1[key] > obj2[key] + self.epsilon:
                return False
            if obj1[key] < obj2[key] - self.epsilon:
                better_in_any = True
        return better_in_any

    def calculate_crowding_distance(self, front):
        if len(front) <= 2:
            for solution in front:
                solution[1]['crowding_distance'] = float('inf')
            return

        for solution in front:
            solution[1]['crowding_distance'] = 0

        for objective in ['energy', 'completion_time']:
            front.sort(key=lambda x: x[1][objective])
            min_obj = front[0][1][objective]
            max_obj = front[-1][1][objective]

            front[0][1]['crowding_distance'] = float('inf')
            front[-1][1]['crowding_distance'] = float('inf')

            if max_obj - min_obj > self.epsilon:
                for i in range(1, len(front) - 1):
                    front[i][1]['crowding_distance'] += \
                        (front[i + 1][1][objective] - front[i - 1][1][objective]) / \
                        (max_obj - min_obj)

    def select_best_solution(self, solutions):
        pareto_front = self.fast_non_dominated_sort(solutions)[0]
        if len(pareto_front) == 1:
            return pareto_front[0]

        self.calculate_crowding_distance(pareto_front)
        return max(pareto_front,
                   key=lambda x: (x[1].get('rank', float('inf')),
                                  x[1].get('crowding_distance', 0)))