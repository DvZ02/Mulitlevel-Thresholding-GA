import copy
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    
    def __init__(self, chromosome=None, hist=None, K_THRESHOLD=None):  # Fix here
        self.chromosome = []
        if hist is not None:
            self.chromosome = sorted(random.sample(range(1, len(hist) - 1), K_THRESHOLD))
        else:
            self.chromosome = sorted(chromosome)
        self.fitness = -1

    def get_chromosome(self):
        return self.chromosome
    
    def get_fitness(self):
        return self.fitness
    
    def set_chromosome(self, chromosome):
        self.chromosome = sorted(chromosome)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def sort_chromosome(self):
        self.chromosome = sorted(self.chromosome)
class GA:
    def __init__(self, params):
        self.POPULATION_SIZE = params["POPULATION_SIZE"]
        self.MAX_GENERATIONS = params["MAX_GENERATIONS"]
        self.CROSSOVER_RATE = params["CROSSOVER_RATE"]
        self.MUTATION_RATE = params["MUTATION_RATE"]
        self.ELITE_SIZE = params["ELITE_SIZE"]
        self.NO_IMPROVEMENT_THRESHOLD = params["NO_IMPROVEMENT_THRESHOLD"]
        self.TOURNAMENT_SIZE = params["TOURNAMENT_SIZE"]
        self.K_THRESHOLD = params["K_THRESHOLD"]
        self.image = cv.imread(params["IMAGE_PATH"], cv.IMREAD_GRAYSCALE)
        self.seed = params["SEED"]
        self.hist = cv.calcHist([self.image], [0], None, [256], [0, 256]).flatten()
        random.seed(self.seed)

    def initialize_population(self):
        population = []
        for _ in range(self.POPULATION_SIZE):
            population.append(Individual(hist=self.hist, K_THRESHOLD=self.K_THRESHOLD))

        return population

    def crossover(self, parent1, parent2):
        if random.random() < self.CROSSOVER_RATE:
            crossover_point = random.randint(0, self.K_THRESHOLD - 1)
            child1 = parent1.get_chromosome()[:crossover_point] + parent2.get_chromosome()[crossover_point:]
            child2 = parent2.get_chromosome()[:crossover_point] + parent1.get_chromosome()[crossover_point:]
            return Individual(chromosome=child1, K_THRESHOLD=self.K_THRESHOLD), Individual(chromosome=child2, K_THRESHOLD=self.K_THRESHOLD)
        return parent1, parent2    

    def mutation(self, individual):
        old_chromosome = copy.deepcopy(individual.get_chromosome())
        for i in range(len(individual.get_chromosome())):
            if random.random() < self.MUTATION_RATE:
                old_chromosome[i] = random.randint(1, len(self.hist) - 1) 
        individual.set_chromosome(old_chromosome)
        return individual
    
    def tournament_selection(self, population):
        tournament = random.sample(population, int(len(population) * self.TOURNAMENT_SIZE))
        best = None
        for t in tournament:
            if best == None or t.get_fitness() > best.get_fitness():
                best = t
        return best

    def otsu_within_class_variance(self, thresholds):
        thresholds = [0] + thresholds + [len(self.hist) - 1]
        total_weight = np.sum(self.hist)
        within_class_variance = 0

        for i in range(len(thresholds) - 1):
            start = thresholds[i]
            end = thresholds[i + 1]
            
            if (end + 1 > 255) is False:
                class_weight = np.sum(self.hist[start:end + 1]) / total_weight
                if class_weight == 0:
                    continue
                
                class_mean = np.sum([j * self.hist[j] for j in range(start, end + 1)]) / np.sum(self.hist[start:end + 1])

                class_variance = np.sum([(j - class_mean) ** 2 * self.hist[j] for j in range(start, end + 1)]) / np.sum(self.hist[start:end + 1])

                class_variance /= 255**2

                within_class_variance += class_weight * class_variance

        return within_class_variance

    def otsu_between_class_variance(self, thresholds):
        thresholds = [0] + thresholds + [len(self.hist) - 1]
        total_mean = np.sum([i * self.hist[i] for i in range(len(self.hist))])
        total_weight = np.sum(self.hist)
        between_class_variance = 0

        for i in range(len(thresholds) - 1):
            start = thresholds[i]
            end = thresholds[i + 1]
            weight = np.sum(self.hist[start:end + 1]) / total_weight
            if weight == 0:
                continue

            if (end + 1 > 255) is False:
                mean = np.sum([j * self.hist[j] for j in range(start, end + 1)]) / np.sum(self.hist[start:end + 1])
                between_class_variance += weight * (mean - total_mean) ** 2

            between_class_variance /= 255**2 

        return between_class_variance

    def otsu_total_class_variance(self, thresholds):
        between = self.otsu_between_class_variance(thresholds)
        within = self.otsu_within_class_variance(thresholds)

        return between ** 2 + within ** 2
    
    def apply_thresholds(self, thresholds):
        thresholds = [0] + thresholds + [256]
        output_image = np.zeros_like(self.image)
        for i in range(len(thresholds) - 1):
            mask = (self.image >= thresholds[i]) & (self.image < thresholds[i + 1])
            output_image[mask] = int(255 / (len(thresholds) - 1)) * i
        return output_image

    def ga(self):
        population = self.initialize_population()
        best_individual = None

        # Initialize fitness and tracking lists
        for individual in population:
            individual.set_fitness(self.otsu_total_class_variance(individual.get_chromosome()))
            if best_individual is None or individual.get_fitness() > best_individual.get_fitness():
                best_individual = individual

        avg_fitness_list = [np.mean([ind.get_fitness() for ind in population])]
        best_fitness_list = [best_individual.get_fitness()]

        plt.ion()
        fig, ax = plt.subplots()
        avg_line, = ax.plot([], [], label="Average Fitness")
        best_line, = ax.plot([], [], label="Best Fitness")
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend()

        for generation in range(self.MAX_GENERATIONS):
            # Elitism
            elite = []
            population = sorted(population, key=lambda x: x.get_fitness(), reverse=True)
            for i in range(int(self.POPULATION_SIZE * self.ELITE_SIZE)):
                elite.append(population[i])

            random.shuffle(population)
            new_population = []
            # Selection, crossover, and mutation
            while len(new_population) + len(elite) < self.POPULATION_SIZE:
                parent1 = self.tournament_selection(elite)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.append(child1)
                new_population.append(child2)

            population = new_population + elite

            # Evaluate fitness
            for individual in population:
                if individual.get_fitness() == -1:
                    individual.set_fitness(self.otsu_total_class_variance(individual.get_chromosome()))
                if individual.get_fitness() > best_individual.get_fitness():
                    best_individual = individual

            # Update fitness tracking
            avg_fitness_list.append(np.mean([ind.get_fitness() for ind in population]))
            best_fitness_list.append(best_individual.get_fitness())

            avg_line.set_data(range(len(avg_fitness_list)), avg_fitness_list)
            best_line.set_data(range(len(best_fitness_list)), best_fitness_list)
            ax.relim()
            ax.autoscale_view(True, True, True)

            plt.draw()
            plt.pause(0.1)

        print("Best individual: ", best_individual.get_chromosome())
        print("Best fitness: ", best_individual.get_fitness())
        cv.imwrite("output.jpg", self.apply_thresholds(best_individual.get_chromosome()))

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot open after the GA finishes
