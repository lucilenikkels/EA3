from src.MOEDA import MOEDA
from src import variation
import matplotlib
from matplotlib import pyplot as plt
import json
from numpy import inf, math
import numpy as np

#print('hypervolumes:', EA.hyperVolumeByGeneration) #print array of hypervolumes
#print('#feval:', EA.numberOfEvaluationsByGeneration) #print array of #feval
#sizes of EA.hyperVolumeByGeneration and EA.numberOfEvaluationsByGeneration are equal

PROBLEMS = [5, 15, 20]


def run_moeda(L, populationSize, use_archive):
    EA = MOEDA(populationSize=populationSize,
               numberOfVariables=L,
               numberOfEvaluations=10 ** 4,
               fitnessFunction='knapsack',
               selection=variation.selection, variation_model=variation.marginalProductModel,
               mutation=variation.mutation,
               tournamentSize=2, mutationProb='auto',
               randomSeed=None, elitistArchive=use_archive)
    EA.evolve()  # Run algorithm
    xs, ys = EA.get_best_front(int(10 ** 4/populationSize)-1)
    return EA.hyperVolumeByGeneration, EA.numberOfEvaluationsByGeneration, xs, ys


def run_reliably(reps=5, L=20, populationSize=100, use_archive=False):
    volumes = []
    evals = []
    x_coords = []
    y_coords = []
    for i in range(reps):
        print("Run "+str(i+1)+" out of "+str(reps)+" (L="+str(L)+", pop="+str(populationSize)+", archive="+str(use_archive)+")")
        vs, es, fx_coords, fy_coords = run_moeda(L, populationSize, use_archive)
        volumes.append(vs)
        evals.append(es)
        x_coords = x_coords + list(fx_coords)
        y_coords = y_coords + list(fy_coords)
    return volumes, evals, x_coords, y_coords


def sort_convergence(evals, fitness):
    evals_sorted = []
    fitness_sorted = []
    while len(evals) > 0:
        ind = evals.index(min(evals))
        evals_sorted.append(evals[ind])
        fitness_sorted.append(fitness[ind])
        del evals[ind]
        del fitness[ind]
    return evals_sorted, fitness_sorted


def summary(p_evals, p_fitness):
    evals, fitness = sort_convergence(p_evals, p_fitness)
    eval_final = []
    fitness_final = []
    stds = []
    start_ind = 0
    counter = 0
    for i in range(len(evals)):
        counter += 1
        if counter == int(len(evals) / 10) or i == len(evals) - 1:
            eval_final.append(np.mean(evals[start_ind:i]))
            mean = np.mean(fitness[start_ind:i])
            fitness_final.append(mean)
            variance = np.mean([(f - mean) ** 2 for f in fitness[start_ind:i]])
            stds.append(math.sqrt(variance))
            start_ind = i
            counter = 0
    return eval_final, fitness_final, stds


def q2a(run=False, name="run.json"):
    if run:
        fitness_archive, evals_archive, xcoords_a, ycoords_a = run_reliably(use_archive=True)
        fitness, evals, xcoords, ycoords = run_reliably()
        results = {"normal": {"fitness": fitness, "evals": evals, "xs": xcoords, "ys": ycoords},
                   "archive": {"fitness": fitness_archive, "evals": evals_archive, "xs": xcoords_a, "ys": ycoords_a}}
        with open("results/a/"+name, "w") as f:
            json.dump(results, f, indent=4)
    else:
        with open("results/a/"+name, "r") as f:
            results = json.load(f)
    plt.figure(0)
    eas = ["normal", "archive"]
    labels = ["W/o archive", "W archive"]
    for i in range(len(eas)):
        evals = [item for sublist in results[eas[i]]["evals"] for item in sublist]
        fitness = [item for sublist in results[eas[i]]["fitness"] for item in sublist]
        x_graph, y_graph, std = summary(evals, fitness)
        plt.plot(x_graph, y_graph, label=labels[i])
        upper = []
        lower = []
        for y in range(len(y_graph)):
            upper.append(y_graph[y]+std[y])
            lower.append(y_graph[y]-std[y])
        plt.fill_between(x_graph, upper, lower, alpha=0.3)
    plt.legend(fontsize=18)
    plt.yscale('log')
    plt.title("Convergence graph for MOEDA (L=20, pop=100)", fontsize=24)
    plt.xlabel("Evaluations", fontsize=20)
    plt.ylabel("Hypervolume", fontsize=20)
    plt.show()


def q2b(run=False, name="run.json"):
    result = {"normal": {L: {"means": [], "sizes": []} for L in PROBLEMS},
              "archive": {L: {"means": [], "sizes": []} for L in PROBLEMS}}
    variants = ["archive"]
    precisions = [0.001, 0.00001]
    for problem in PROBLEMS:
        plt.figure()
        for idx, ea in enumerate(variants):
            precision = 0.001 # precisions[idx]
            plt.title(ea+": L="+str(problem))
            plt.yscale('log')
            last_val = -1.0
            res = 0.0
            size = 4
            if run:
                while abs(res-last_val) > precision and size != 9999:
                    last_val = res
                    size = min(size*2, 9999)
                    fitness, evals, _, _ = run_reliably(5, problem, size, ea == "archive")
                    result[ea][problem]["sizes"].append(size)
                    res = np.mean(fitness)
                    result[ea][problem]["means"].append(res)
                    with open("results/b/"+name, "w") as f:
                        json.dump(result, f, indent=4)
                prev = result[ea][problem]["means"][-1]
                if abs(prev-result[ea][problem]["means"][-2]) < 0.0000000000001:
                    ub = result[ea][problem]["sizes"][-2]
                    lb = result[ea][problem]["sizes"][-3]
                    print("They are the same")
                else:
                    ub = result[ea][problem]["sizes"][-1]
                    lb = result[ea][problem]["sizes"][-2]
                while ub-lb > 2:
                    size = int((ub+lb)/2)
                    fitness, evals, _, _ = run_reliably(5, problem, size, ea == "archive")
                    res = np.mean(fitness)
                    result[ea][problem]["sizes"].append(size)
                    result[ea][problem]["means"].append(res)
                    if res >= prev-precision:
                        prev = res
                        ub = size
                    else:
                        lb = size
                    with open("results/b/"+name, "w") as f:
                        json.dump(result, f, indent=4)
            else:
                with open("results/b/" + name, "r") as f:
                    result = json.load(f)
            # plt.scatter(result[ea][str(problem)]["sizes"], result[ea][str(problem)]["means"], label=ea)
        plt.legend()
    plt.show()


def q2c(name="run.json"):
    return 0


#q2a(True, "run06.json")
q2b(True, "run16.json")
#run_moeda(20, 100, True)

# 11 contains correct version of L=5 without archive (p=0.001)
# 15 contains correct versions of L=10, L=20 without archive (p=0.001)

