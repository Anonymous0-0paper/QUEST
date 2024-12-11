import random

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class NSGA3(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.population_size = 20
        self.max_iterations = 100
        self.mutation_rate = 0.1

    def run(self):
        population = self.initialize_population()
        best_assignment = None
        best_objectives = float('inf')

        for iteration in range(self.max_iterations):
            new_population = []

            for _ in range(self.population_size):
                parents = self.select_parents(population, k=2)
                child = self.crossover(parents[0], parents[1])
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

            for individual in population:
                self.assign = individual
                objectives = self.calculate_objectives()
                total_objective = sum(objectives.values())

                if total_objective < best_objectives:
                    best_objectives = total_objective
                    best_assignment = individual.copy()

        self.assign = best_assignment
        super().run()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for task_id in range(len(self.dag.subtasks)):
                individual[task_id] = random.randrange(len(self.network.nodes))
            population.append(individual)
        return population

    def calculate_objectives(self):
        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        return {
            'energy': energy / 1000,
            'completion_time': completion_time
        }

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            task_id = random.randrange(len(self.dag.subtasks))
            individual[task_id] = random.randrange(len(self.network.nodes))
        return individual

    def crossover(self, parent1, parent2):
        child = {}
        for task_id in range(len(self.dag.subtasks)):
            if random.random() < 0.5:
                child[task_id] = parent1[task_id]
            else:
                child[task_id] = parent2[task_id]
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
        for key in ['energy', 'completion_time']:
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

        for objective in ['energy', 'completion_time']:
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
        evaluated_solutions = []
        for individual in population:
            self.assign = individual
            objectives = self.calculate_objectives()
            evaluated_solutions.append((individual, objectives))

        fronts = self.fast_non_dominated_sort(evaluated_solutions)
        for front in fronts:
            self.calculate_crowding_distance(front)

        selected_parents = []
        for _ in range(k):
            candidates = random.sample(evaluated_solutions, k=3)
            winner = min(candidates, key=lambda x: (
                x[1].get('rank', float('inf')),
                -x[1].get('crowding_distance', 0)
            ))
            selected_parents.append(winner[0])

        return selected_parents
