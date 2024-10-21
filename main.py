import ga as GA
import time
import asyncio

def main(params):
    ga = GA.GA(params)
    ga.ga()


if __name__ == "__main__":
    params = [
            {
                "POPULATION_SIZE": 250, "MAX_GENERATIONS": 150, "CROSSOVER_RATE": 0.75, "MUTATION_RATE": 0.3, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 8, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/12.jpg", 
                "SEED": 0, "K_THRESHOLD": 2
            }
        ]
    seed_str = input("Enter a seed (or press Enter to use a random seed): ")

    if(seed_str.strip()):
        params[0]["SEED"] = int(seed_str)
    else:
        params[0]["SEED"] = int(time.time())
        
    k_threshold = input("Enter the number of thresholds (k - default = 2): ")
    
    if(k_threshold.strip() == ""):
        params[0]["K_THRESHOLD"] = 2
    else:
        params[0]["K_THRESHOLD"] = int(k_threshold)
 
    main(params[0])