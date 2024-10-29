import ga as GA
import time
import cv2 as cv
import matplotlib.pyplot as plt

def main(params):

    best_runs_image = None
    best_runs_fitness = 0
    best_individual = None
    for _ in range(10):
        params["SEED"] = int(time.time())
        ga = GA.GA(params)
        obj = ga.ga()

        if best_individual == None or obj['best_individual'].get_fitness() > best_runs_fitness:
            best_runs_image = obj['output_image']
            best_runs_fitness = obj['best_individual'].get_fitness()
            best_individual = obj['best_individual']

    print("Best individual: ", best_individual.get_chromosome())
    print("Best fitness: ", best_individual.get_fitness())
    cv.imwrite("output.jpg", best_runs_image)

    plt.figure()
    plt.hist(cv.imread(params["IMAGE_PATH"]).ravel(), 256, [1, 256])
    for threshold in best_individual.get_chromosome():
        plt.axvline(x=threshold, color='r')
    plt.title("Histogram of pixel intensities with thresholds marked")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.show()

    

    plt.ioff()
    plt.show()




if __name__ == "__main__":
    params = [
            {
                "POPULATION_SIZE": 50, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.8, "MUTATION_RATE": 0.1, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 8, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/Medical images/022.png", 
                "SEED": 0, "K_THRESHOLD": 2, "FITNESS_FUNCTION": "kapur_entropy"
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