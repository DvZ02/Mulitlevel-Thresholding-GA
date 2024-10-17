import copy
import random

class GA:
    def __init__(self, params, seed):
        self.POPULATION_SIZE = params["POPULATION_SIZE"]
        self.MAX_GENERATIONS = params["MAX_GENERATIONS"]
        self.CROSSOVER_RATE = params["CROSSOVER_RATE"]
        self.MUTATION_RATE = params["MUTATION_RATE"]
        self.ELITE_SIZE = params["ELITE_SIZE"]
        # self.NO_IMPROVEMENT_THRESHOLD = params["NO_IMPROVEMENT_THRESHOLD"]
        self.TOURNAMENT_SIZE = params["TOURNAMENT_SIZE"]
        self.CHROMOSOME_LENGTH = params["CHROMOSOME_LENGTH"]
        self.seed = seed
        random.seed(self.seed)

    def initialize_population(self):
        population = []
        for _ in range(self.POPULATION_SIZE):
            chromosome = [random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in range(self.CHROMOSOME_LENGTH)]
            population.append(chromosome)
        return population

    def crossover(self, parent1, parent2):
        if random.random() < self.CROSSOVER_RATE:
            crossover_point = random.randint(0, self.CHROMOSOME_LENGTH - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2    

    def mutation(self, chromosome):
        mutated_chromosome = copy.deepcopy(chromosome)
        for i in range(len(mutated_chromosome)):
            if random.random() < self.MUTATION_RATE:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        return mutated_chromosome
    
    def elitism(self, population):
        return None
    
    def tournemant_selection(self, population, fitness):
        tournament = random.sample(population, self.TOURNAMENT_SIZE)
        best = 0
        best_index = 0
        for t in tournament:
            temp_fit = fitness[population.index(t)]
            if temp_fit > best:
                best = temp_fit
                best_index = tournament.index(t)

        return tournament[best_index]

    def fitness(self, chromosome):
        return None # Placeholder
    
    

    def ga(self):
        population = self.initialize_population()

        return None # Placeholder