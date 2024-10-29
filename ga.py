import copy
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

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
        self.image = cv.imread(params["IMAGE_PATH"])
        self.seed = params["SEED"]
        self.hist = cv.calcHist([self.image], [0], None, [256], [0, 256]).flatten()
        random.seed(self.seed)
        self.mutationOps = [0, 1, 2]
        self.currentMutationOp = 0
        self.fitness_cache = {}
        self.fitness_function = params["FITNESS_FUNCTION"]

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
        
    # Implement new mutation
    def hybridMutation(self, individual):
        if random.random() < self.MUTATION_RATE:
            if self.currentMutationOp == 0:
                return self.ILS_1(individual)
            elif self.currentMutationOp == 1:
                return self.ILS_2(individual)
            elif self.currentMutationOp == 2:
                return self.Tabu(individual)
            # elif self.currentMutationOp == 3:
            #     return self.mutation(individual)     

    def VNS(self, elite):
        new_elite = []
        for ind in elite:
            if self.currentMutationOp == 0:
                new_elite.append(self.ILS_1(ind))
            elif self.currentMutationOp == 1:
                new_elite.append(self.ILS_2(ind))
            elif self.currentMutationOp == 2:
                new_elite.append(self.Tabu(ind))
            # elif self.currentMutationOp == 3:
            #     elite[elite.index(ind)] = self.mutation(ind) 
        return new_elite  
    

    def Tabu(self, individual, tabu_size=5, max_iter=30, neighbourhood_size=10, ls_iterations = 5):
        best_individual = individual
        current_individual = copy.deepcopy(individual)
        tabu_list = []

        for _ in range(max_iter):
            neighbours = []

            for _ in range(neighbourhood_size):
                neighbour = copy.deepcopy(current_individual)
                neighbour = self.local_search(neighbour, ls_iterations, ILS = random.choice([True, False]))
                neighbours.append(neighbour)

            for neighbour in neighbours:
                self.calculate_fitness_wrapper(neighbour)

            if self.fitness_function == 'otsu_within_class_variance' or self.fitness_function == 'otsu_total_class_variance' or self.fitness_function == 'kapur_entropy':
                neighbours = sorted(neighbours, key=lambda x: x.get_fitness(), reverse=True)
            else:
                neighbours = sorted(neighbours, key=lambda x: x.get_fitness())
            best_neighbour = None
            for neighbour in neighbours:
                NOT_IN_TABU =True
                for tabu in tabu_list:
                    if neighbour.get_chromosome() == tabu.get_chromosome():
                        NOT_IN_TABU = False
                        break

                if NOT_IN_TABU:
                    best_neighbour = neighbour
                    tabu_list.append(best_neighbour)
                    if(len(tabu_list) > tabu_size):
                        tabu_list.pop(0)
                    break
                    
            if self.better_fitness(best_neighbour, current_individual):
                current_individual = best_neighbour
                if self.better_fitness(current_individual, best_individual):
                    best_individual = copy.deepcopy(current_individual)
    
        return best_individual
                
    
    def ILS_1(self, individual, ls_iterations = 5, num_iterations = 30):

        best_individual = self.local_search(individual, ls_iterations, ILS = False)

        for iteration in range(num_iterations):

            perturbed_solution = self.mutation(best_individual)

            improved_solution = self.local_search(perturbed_solution,  ls_iterations, ILS = False) 

            if self.better_fitness(improved_solution, best_individual):
                best_individual = improved_solution
        
    

        return best_individual
    
    def ILS_2(self, individual, ls_iterations = 5, num_iterations = 30):

        best_individual = self.local_search(individual, ls_iterations, ILS = True)

        for iteration in range(num_iterations):

            perturbed_solution = self.mutation(best_individual)
            improved_solution = self.local_search(perturbed_solution, ls_iterations, ILS = True)

            if self.better_fitness(improved_solution, best_individual):
                best_individual = improved_solution
        
    

        return best_individual

    def local_search(self, solution, iterations, ILS):
        best_solution = copy.deepcopy(solution)
        self.calculate_fitness_wrapper(best_solution)

        for _ in range(iterations):
            neighbour = copy.deepcopy(best_solution)
            randInd = random.randint(0, len(neighbour.get_chromosome()) - 1)
            incDec = random.choice([True, False])

            if ILS:
                adjustment = 1
            else:
                adjustment = 5

            if incDec:
                neighbour.get_chromosome()[randInd] += adjustment
            else:
                neighbour.get_chromosome()[randInd] -= adjustment
            
            neighbour.set_chromosome([min(max(gene, 0), 255) for gene in neighbour.get_chromosome()])

            self.calculate_fitness_wrapper(neighbour)

            if self.better_fitness(neighbour, best_solution):
                return neighbour

        return best_solution
    
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
            if best == None or self.better_fitness(t, best):
                best = t
        return best

    def otsu_within_class_variance(self, thresholds):
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
        
        fitness_value = between ** 2 + within ** 2
        
        return fitness_value
    
    def kapur_entropy(self, thresholds):
        total_pixels = np.sum(self.hist)
        
        # Normalize the histogram (compute probabilities)
        probabilities = self.hist / total_pixels
        
        total_entropy = 0.0
        
        # Iterate through regions defined by thresholds
        for i in range(len(thresholds) + 1):
            region_entropy = 0.0
            region_sum = 0.0
            
            # Define region boundaries
            start = 0 if i == 0 else thresholds[i - 1] + 1  # Start of region
            end = len(probabilities) - 1 if i == len(thresholds) else thresholds[i]  # End of region
            
            # Calculate omega (sum of probabilities for the region)
            for t in range(start, end + 1):
                region_sum += probabilities[t]
            
            # Calculate the entropy for the region
            for t in range(start, end + 1):
                if probabilities[t] > 0 and region_sum > 0:
                    region_entropy -= (probabilities[t] / region_sum) * np.log(probabilities[t] / region_sum)
            
            # Add the region entropy to the total entropy
            total_entropy += region_entropy

        return total_entropy
    
    def calculate_fitness(self, thresholds):
        chromosome_key = tuple(thresholds)
        
        if chromosome_key in self.fitness_cache:
            return self.fitness_cache[chromosome_key]
        
        fitness_value = None
        if self.fitness_function == "otsu_total_class_variance":
            fitness_value = self.otsu_total_class_variance(thresholds)
        elif self.fitness_function == "otsu_between_class_variance":
            fitness_value = self.otsu_between_class_variance(thresholds)
        elif self.fitness_function == "otsu_within_class_variance":
            fitness_value = self.otsu_within_class_variance(thresholds)
        elif self.fitness_function == "kapur_entropy":
            fitness_value = self.kapur_entropy(thresholds)
        
        self.fitness_cache[chromosome_key] = fitness_value

        return fitness_value
    
    def apply_thresholds(self, thresholds):
        thresholds = sorted(thresholds)
        # Add boundaries to the thresholds list
        output_image = np.zeros_like(self.image)
        
        # Compute intensity levels to assign for each thresholded region
        num_levels = len(thresholds) - 1
        intensity_step = 255 // (num_levels - 1)
        
        # Apply each threshold range as a mask
        for i in range(num_levels):
            mask = (self.image >= thresholds[i]) & (self.image < thresholds[i + 1])
            output_image[mask] = thresholds[i]
        
        return output_image

        # Define a helper function for fitness calculation
    def calculate_fitness_wrapper(self, individual):
        if individual.get_fitness() == -1:  # Only evaluate if fitness is not set
            individual.set_fitness(self.calculate_fitness(individual.get_chromosome()))
        return individual
    
    def better_fitness(self, ind_1, ind_2):
        if ind_1 == None:
            return False
        if ((self.fitness_function == 'otsu_total_class_variance' or self.fitness_function == 'otsu_between_class_variance' or self.fitness_function == 'kapur_entropy') and ind_1.get_fitness() > ind_2.get_fitness()) or (self.fitness_function == 'otsu_within_class_variance' and ind_1.get_fitness() < ind_2.get_fitness()):
            return True
        return False

    # Inside the GA class's ga() function
    def ga(self):
        population = self.initialize_population()
        best_individual = None

        # Initialize fitness and tracking lists
        for individual in population:
            self.calculate_fitness_wrapper(individual)
            if best_individual == None or self.better_fitness(individual, best_individual):
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

        same_best_counter = 0

        # Generations loop with threading for fitness evaluation
        same_best_counter = 0
        for generation in range(self.MAX_GENERATIONS):
            # Elitism
            if self.fitness_function == 'otsu_within_class_variance' or self.fitness_function == 'otsu_total_class_variance' or self.fitness_function == 'kapur_entropy':
                elite = sorted(population, key=lambda x: x.get_fitness(), reverse=True)[:int(self.POPULATION_SIZE * self.ELITE_SIZE)]
            else:
                elite = sorted(population, key=lambda x: x.get_fitness())[:int(self.POPULATION_SIZE * self.ELITE_SIZE)]

            if same_best_counter == self.NO_IMPROVEMENT_THRESHOLD:
                # average of elite before
                avg = 0
                for ind in elite:
                    avg += ind.get_fitness()
                avg /= len(elite)

                print("Average before: ", avg)
                self.currentMutationOp = random.randint(0, 2)
                elite = self.VNS(elite)
                same_best_counter = 0

                avg = 0
                for ind in elite:
                    avg += ind.get_fitness()
                avg /= len(elite)
                print("Average after: ", avg)

            new_population = []
            while len(new_population) + len(elite) < self.POPULATION_SIZE:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                if child1:
                    new_population.append(child1)
                if child2:
                    new_population.append(child2)

            population = elite + new_population

            # Threaded fitness evaluation
            with ThreadPoolExecutor() as executor:
                updated_population = list(executor.map(lambda ind: self.calculate_fitness_wrapper(ind), population))

            # Update the population with the newly calculated fitness values
            population = updated_population

            # Determine the best individual
            prev_best_individual = best_individual
            for individual in population:
                if self.better_fitness(individual, best_individual):
                    best_individual = individual

            if prev_best_individual == best_individual:
                same_best_counter += 1
            else:
                print("Generation: ", generation, "Best fitness: ", best_individual.get_fitness(), best_individual.get_chromosome())
                same_best_counter = 0

            # Update fitness tracking
            avg_fitness_list.append(np.mean([ind.get_fitness() for ind in population]))
            best_fitness_list.append(best_individual.get_fitness())

            avg_line.set_data(range(len(avg_fitness_list)), avg_fitness_list)
            best_line.set_data(range(len(best_fitness_list)), best_fitness_list)
            ax.relim()
            ax.autoscale_view(True, True, True)

            plt.draw()
            plt.pause(0.1)

        output_image = self.apply_thresholds(best_individual.get_chromosome())

        plt.close()
        return {"best_individual": best_individual, "output_image": output_image}
