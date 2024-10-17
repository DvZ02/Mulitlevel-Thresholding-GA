import ga as GA
import time
import asyncio

def main(params, seed):
    ga = GA.GA(params, seed)
    ga.ga()


if __name__ == "__main__":
    params = [
            {
                "POPULATION_SIZE": 100, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.7, "MUTATION_RATE": 0.05, 
                "ELITE_SIZE": 3, "NO_IMPROVEMENT_THRESHOLD": 8, "TOURNAMENT_SIZE": 3, "CHROMOSOME_LENGTH": 256	
            }
            #Can add more if needed for tests
        ]
    seed_str = input("Enter a seed (or press Enter to use a random seed): ")
    seed = 0

    if(seed_str.strip()):
        seed = int(seed_str)
    else:
        seed = int(time.time())
 
    main(params[0], seed)