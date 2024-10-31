import os
import ga as GA
import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def save_output(param, thresholds, image, image_name, level, run):
    regions = np.digitize(image, bins=thresholds)

    plt.imsave(f"./output/{param}/{run}/{level}/{image_name}", regions, cmap='gray')
    # produced = cv.imread(f"./output/{image_name}_{level}.png", cv.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))

    ax[0].imshow(image, cmap='gray') 
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image.ravel(), 256, [1, 256])
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    ax[2].imshow(regions, cmap='Accent')
    ax[2].set_title('Result')
    ax[2].axis('off')

    plt.subplots_adjust()
    plt.savefig(f"./output/{param}/{run}/{level}/plot_{image_name}")

def main(params, img_name, run, param):

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
    save_output(param, best_individual.get_chromosome(), cv.imread(params["IMAGE_PATH"], cv.IMREAD_GRAYSCALE), img_name, params["K_THRESHOLD"], run)
    # cv.imwrite("output.jpg", best_runs_image)

    # plt.figure()
    # plt.hist(cv.imread(params["IMAGE_PATH"]).ravel(), 256, [1, 256])
    # for threshold in best_individual.get_chromosome():
    #     plt.axvline(x=threshold, color='r')
    # plt.title("Histogram of pixel intensities with thresholds marked")
    # plt.xlabel("Pixel intensity")
    # plt.ylabel("Frequency")
    # plt.show()
    # plt.ioff()
    # plt.show()



if __name__ == "__main__":
    params = [
            {
                "POPULATION_SIZE": 50, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.8, "MUTATION_RATE": 0.1, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 8, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/Medical images/022.png", 
                "SEED": 0, "K_THRESHOLD": 2, "FITNESS_FUNCTION": "kapur_entropy"
            },
            {
                "POPULATION_SIZE": 100, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.8, "MUTATION_RATE": 0.1, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 8, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/Medical images/022.png", 
                "SEED": 0, "K_THRESHOLD": 2, "FITNESS_FUNCTION": "kapur_entropy"
            },
            {
                "POPULATION_SIZE": 50, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.6, "MUTATION_RATE": 0.3, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 10, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/Medical images/022.png", 
                "SEED": 0, "K_THRESHOLD": 2, "FITNESS_FUNCTION": "kapur_entropy"
            },
            {
                "POPULATION_SIZE": 100, "MAX_GENERATIONS": 100, "CROSSOVER_RATE": 0.7, "MUTATION_RATE": 0.25, 
                "ELITE_SIZE": 0.3, "NO_IMPROVEMENT_THRESHOLD": 6, "TOURNAMENT_SIZE": 0.15, "IMAGE_PATH": "./images/Medical images/022.png", 
                "SEED": 0, "K_THRESHOLD": 2, "FITNESS_FUNCTION": "kapur_entropy"
            }
        ]
    seed_str = input("Enter a seed (or press Enter to use a random seed): ")

    image_names = ["022.png", "032.png", "042.png",
                   "052.png", "062.png", "072.png",
                   "082.png", "092.png", "102.png",
                   "112.png"
                   ]

    if(seed_str.strip()):
        params[0]["SEED"] = int(seed_str)
    else:
        params[0]["SEED"] = int(time.time())
        
    k_threshold = input("Enter the number of thresholds (k - default = 2): ")
    
    if(k_threshold.strip() == ""):
        params[0]["K_THRESHOLD"] = 2
    else:
        params[0]["K_THRESHOLD"] = int(k_threshold)
    
    for param in range(0, 4):
        print(f"====== Running for parameter set: {param+1} ======")
        for run in range(1, 11):
            print(f"====== Run: {run} ======")
            for i in range(0, 10):
                print(f"====== Running level k=2 to k=5 for Image: {image_names[i]} ======")
                params[param]["IMAGE_PATH"] = f"./images/Medical images/{image_names[i]}"
                for j in range(2, 6):
                    params[param]["K_THRESHOLD"] = j
                    main(params[param], img_name=image_names[i], run=run, param=param+1)

    # main(params[0])
    # img = cv.imread(params[0]["IMAGE_PATH"], cv.IMREAD_GRAYSCALE)
    # save_output([93, 183], img, "output", params[0]["K_THRESHOLD"])

    # for i in range(0, 4):
    #     directory_path = f"./output/{i+1}/"
    #     os.makedirs(directory_path, exist_ok=True)
    #     for j in range(1, 11):
    #         directory_path = f"./output/{i+1}/{j}"
    #         os.makedirs(directory_path, exist_ok=True)
    #         for k in range(2, 6):
    #             directory_path = f"./output/{i+1}/{j}/{k}"
    #             os.makedirs(directory_path, exist_ok=True)


