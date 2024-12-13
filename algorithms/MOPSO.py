import random
import numpy as np
from algorithms.Algorithm import Algorithm
from model.DAGModel import DAGModel
from network.Network import Network


class MOPSO(Algorithm):
    def __init__(self, network: Network, dag: DAGModel):
        super().__init__(network, dag)
        self.individual_cognition = 1.5
        self.social_communication = 1.5
        self.constriction_factor = 2.0
        self.inertia_weight = 1.0
        self.population_size = 20
        self.max_iterations = 50
        self.particle_dim = len(self.dag.subtasks)
        self.best_solution = None

    def run(self):
        self.best_solution = self.mopso_scheduling()
        self.assign = self.best_solution
        super().run()

    def initialize_particles(self):
        particles = []
        velocities = []
        num_nodes = len(self.network.nodes)

        for _ in range(self.population_size):
            particle = {}
            for task_id in range(self.particle_dim):
                particle[task_id] = random.randint(0, num_nodes - 1)

            velocity = {}
            for task_id in range(self.particle_dim):
                velocity[task_id] = random.uniform(-1, 1)

            particles.append(particle)
            velocities.append(velocity)

        return particles, velocities

    def evaluate_particle(self, particle):
        self.assign = particle
        super().run()

        objectives = {
            'energy': self.calculate_energy(),
            'completion_time': self.calculate_completion_time(),
            'dominated_solutions': [],
            'domination_count': 0,
            'crowding_distance': 0
        }
        return objectives

    def dominates(self, obj1, obj2):
        better_in_any = False
        for key in ['energy', 'completion_time']:
            if obj1[key] > obj2[key] + 1e-10:
                return False
            if obj1[key] < obj2[key] - 1e-10:
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

            if max_obj - min_obj > 1e-10:
                for i in range(1, len(front) - 1):
                    front[i][1]['crowding_distance'] += (
                                                                front[i + 1][1][objective] - front[i - 1][1][objective]
                                                        ) / (max_obj - min_obj)

    def mopso_scheduling(self):
        particles, velocities = self.initialize_particles()
        pbest = particles.copy()
        pbest_objectives = [self.evaluate_particle(p) for p in pbest]
        gbest = particles[0].copy()
        gbest_objectives = pbest_objectives[0]

        for iteration in range(self.max_iterations):
            particle_objectives = [self.evaluate_particle(p) for p in particles]

            for i in range(self.population_size):
                if self.dominates(particle_objectives[i], pbest_objectives[i]):
                    pbest[i] = particles[i].copy()
                    pbest_objectives[i] = particle_objectives[i].copy()

            non_dominated = []
            for i in range(self.population_size):
                dominated = False
                for j in range(self.population_size):
                    if i != j and self.dominates(particle_objectives[j], particle_objectives[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append((particles[i], particle_objectives[i]))

            self.calculate_crowding_distance(non_dominated)
            gbest_candidate = max(non_dominated,
                                  key=lambda x: x[1]['crowding_distance'])

            if self.dominates(gbest_candidate[1], gbest_objectives):
                gbest = gbest_candidate[0].copy()
                gbest_objectives = gbest_candidate[1].copy()

            for i in range(self.population_size):
                r1 = random.random()
                r2 = random.random()

                for task_id in range(self.particle_dim):
                    velocities[i][task_id] = (
                            self.inertia_weight * velocities[i][task_id] +
                            self.individual_cognition * r1 * (pbest[i][task_id] - particles[i][task_id]) +
                            self.social_communication * r2 * (gbest[task_id] - particles[i][task_id])
                    )

                    velocities[i][task_id] *= self.constriction_factor

                    new_pos = int(particles[i][task_id] + velocities[i][task_id])
                    particles[i][task_id] = max(0, min(new_pos, len(self.network.nodes) - 1))

        return gbest