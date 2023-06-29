import numpy as np
from Individual import Individual
from copy import deepcopy

class Genetic:
    def __init__(self, target_individual: Individual,
                        population_size: np.int32,
                        tournament_size: np.float32,
                        parents_size: np.float32) -> None:
        
        self.target = target_individual

        self.size = self.target.size
        self.population_size = population_size
        self.tournament_size = int(tournament_size*population_size)
        self.parents_size = int(parents_size*population_size)

        self.population: list
        self.best_parents: list
        
    def initialize_population(self) -> None:
        self.population = self.population_size * [Individual('', self.size)]
    
    def selection(self) -> None:
        self.best_parents = [None] * self.parents_size
        for i in range(self.parents_size):
            cands_idx = np.random.choice(range(self.population_size), self.tournament_size)
            cands = self.population[cands_idx]
            best_cand, best_fit = cands[0], self.target.fitness(cands[0].img)
            for cand in cands[1:]:
                cand_fitness = self.target.fitness(cand.img)
                if cand_fitness < best_fit:
                    best_fit = cand_fitness
                    best_cand = cand
                    
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