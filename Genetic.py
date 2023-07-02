import numpy as np
from Individual import Individual
from copy import deepcopy


class Genetic:
    def __init__(self, target_individual: Individual,
                        population_size: int,
                        tournament_size: float,
                        parents_size: float,
                        mutation_rate: float,
                        blend_use: float,
                        block_size: tuple,
                        num_blocks_initialize: tuple,
                        max_blocks_mutation: int) -> None:
        
        self.target = target_individual

        self.size = self.target.size
        self.population_size = population_size
        self.tournament_size = int(tournament_size*population_size)
        self.parents_size = int(parents_size*population_size)
        self.new_generation_size = self.population_size - self.parents_size
        self.mutation_rate = mutation_rate
        self.blend_use = blend_use
        self.block_size = block_size
        self.num_blocks_initialize = num_blocks_initialize
        self.max_blocks_mutation = max_blocks_mutation

        self.population: list
        self.best_parents: list
        
    def initialize_population(self) -> None:
        self.population = [Individual('', self.size, self.block_size,
                                            self.num_blocks_initialize)
                            for _ in range(self.population_size)]
    
    def selection(self) -> None:
        self.best_parents = [None] * self.parents_size
        for i in range(self.parents_size):
            cands_idx = np.random.choice(range(self.population_size), self.tournament_size)
            best_cand = self.population[cands_idx[0]]
            best_fit = self.target.fitness(self.population[cands_idx[0]].img)

            for idx in cands_idx[1:]:
                cand_fitness = self.target.fitness(self.population[idx].img)
                if cand_fitness < best_fit:
                    best_fit = cand_fitness
                    best_cand = self.population[idx]
                    
            self.best_parents[i] = best_cand

    def crossover_blend(self, parent_1: Individual, parent_2: Individual) -> Individual:
        x = np.random.random()
        child = deepcopy(parent_1)
        child.img = x*parent_1.img + (1-x)*parent_2.img
        
        return child

    def crossover_two_point(self, parent_1: Individual, parent_2: Individual) -> Individual:
        point1 = np.random.choice(range(self.size[0]))
        point2 = np.random.choice(range(self.size[1]))

        child = deepcopy(parent_1)
        child.img[:point1, :point2] = parent_1.img[:point1, :point2]
        child.img[:point1, point2:] = parent_2.img[:point1, point2:]
        child.img[point1:, :point2] = parent_2.img[point1:, :point2]
        child.img[point1:, point2:] = parent_1.img[point1:, point2:]
        
        return child

    def top_individuals(self) -> tuple:
        top_individual: Individual
        top_fitness = 10**9
        for i, individual in enumerate(self.population):
            if (f := self.target.fitness(individual.img)) < top_fitness:
                top_fitness = f
                top_individual = self.population[i]

        return top_individual, top_fitness

    def run(self, generations: int, print_every: int = 50) -> None:
        for i in range(generations):
            self.selection()
            parent_1_idxs = np.random.permutation(self.new_generation_size) % self.parents_size
            parent_2_idxs = np.random.permutation(self.new_generation_size) % self.parents_size
            for j, (p1, p2) in enumerate(zip(parent_1_idxs, parent_2_idxs)):
                if np.random.random() <= self.blend_use:
                    self.population[j] = self.crossover_blend(
                        self.best_parents[p1], self.best_parents[p2])
                else:
                    self.population[j] = self.crossover_two_point(
                        self.best_parents[p1], self.best_parents[p2])
                self.population[j].mutate(self.mutation_rate, self.max_blocks_mutation)
            
            self.population[self.new_generation_size:] = self.best_parents

            if (i + 1) % print_every == 0:
                print(f'Generation {i+1}...')