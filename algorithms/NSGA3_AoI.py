import random

from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class NSGA3_AoI(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.population_size = 20
        self.max_iterations = 100
        self.mutation_rate = 0.1
        self.per_iteration_objectives: list[float] = []

    def run(self):
        population = self.initialize_population()
        self.assign = population[0]
        super().run()

        best_assignment = None
        best_objectives = float('inf')

        for _ in range(self.max_iterations):
            population = self.create_next_generation(population)

            for individual in population:
                self.assign = individual
                super().run()
                objectives = self.calculate_objectives()
                total_objective = sum(objectives.values())

                if total_objective < best_objectives:
                    best_objectives = total_objective
                    best_assignment = individual.copy()

            self.per_iteration_objectives.append(best_objectives)

        self.assign = best_assignment

    def initialize_population(self):
        num_tasks = len(self.dag.subtasks)
        num_nodes = len(self.network.nodes)
        return [
            {task_id: random.randrange(num_nodes) for task_id in range(num_tasks)}
            for _ in range(self.population_size)
        ]

    def calculate_objectives(self):
        energy = self.calculate_energy()
        completion_time = self.calculate_completion_time()
        data_age = self.calculate_data_age()

        # Aggregate objectives with weights
        weights = {
            'energy': 0.5,
            'completion_time': 0.3,
            'data_age': 0.2
        }
        return {
            'energy': (energy / 1000) * weights['energy'],
            'completion_time': completion_time * weights['completion_time'],
            'data_age': (data_age * 1000) * weights['data_age']
        }

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            task_id = random.randrange(len(self.dag.subtasks))
            individual[task_id] = random.randrange(len(self.network.nodes))
        return individual

    def crossover(self, parent1, parent2):
        return {
            task_id: (parent1[task_id] if random.random() < 0.5 else parent2[task_id])
            for task_id in range(len(self.dag.subtasks))
        }

    def fast_non_dominated_sort(self, solutions):
        domination_count = {id(s): 0 for s in solutions}
        dominated_solutions = {id(s): [] for s in solutions}
        fronts = [[]]

        for p in solutions:
            p_dominates = dominated_solutions[id(p)]
            for q in solutions:
                if p is not q:
                    if self.dominates(p[1], q[1]):
                        p_dominates.append(q)
                    elif self.dominates(q[1], p[1]):
                        domination_count[id(p)] += 1

            if domination_count[id(p)] == 0:
                p[1]['rank'] = 0
                fronts[0].append(p)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for p in fronts[current_front]:
                for q in dominated_solutions[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q[1]['rank'] = current_front + 1
                        next_front.append(q)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]

    def dominates(self, obj1, obj2):
        return (
                all(obj1[key] <= obj2[key] for key in ['energy', 'completion_time']) and
                any(obj1[key] < obj2[key] for key in ['energy', 'completion_time'])
        )

    def calculate_crowding_distance(self, front):
        if len(front) <= 2:
            for solution in front:
                solution[1]['crowding_distance'] = float('inf')
            return

        num_objectives = len(front[0][1])
        for solution in front:
            solution[1]['crowding_distance'] = 0

        for objective in ['energy', 'completion_time']:
            front.sort(key=lambda x: x[1][objective])
            front[0][1]['crowding_distance'] = float('inf')
            front[-1][1]['crowding_distance'] = float('inf')
            obj_range = front[-1][1][objective] - front[0][1][objective] or 1e-9

            for i in range(1, len(front) - 1):
                front[i][1]['crowding_distance'] += (
                        (front[i + 1][1][objective] - front[i - 1][1][objective]) / obj_range
                )

    def select_parents(self, population, k=2):
        evaluated_solutions = [(individual, self.calculate_objectives()) for individual in population]
        fronts = self.fast_non_dominated_sort(evaluated_solutions)

        for front in fronts:
            self.calculate_crowding_distance(front)

        selected = []
        for _ in range(k):
            tournament = random.sample(evaluated_solutions, 3)
            winner = min(
                tournament, key=lambda x: (
                    x[1].get('rank', float('inf')),
                    -x[1].get('crowding_distance', 0)
                )
            )
            selected.append(winner[0])

        return selected

    def create_next_generation(self, population):
        new_population = []
        for _ in range(self.population_size):
            parents = self.select_parents(population, k=2)
            child = self.crossover(parents[0], parents[1])
            new_population.append(self.mutation(child))
        return new_population
