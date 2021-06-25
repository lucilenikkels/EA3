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

PROBLEMS = [5, 10, 20]


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


def q2a(run=False, name="run.json"):
	if run:
		fitness, evals, xcoords, ycoords = run_reliably()
		fitness_archive, evals_archive, xcoords_a, ycoords_a = run_reliably(use_archive=True)
		results = {"normal": {"fitness": fitness, "evals": evals, "xs": xcoords, "ys": ycoords},
				   "archive": {"fitness": fitness_archive, "evals": evals_archive, "xs": xcoords_a, "ys": ycoords_a}}
		with open("results/a/"+name, "w") as f:
			json.dump(results, f, indent=4)
	else:
		with open("results/a/"+name, "r") as f:
			results = json.load(f)
	plt.figure(0)
	plt.scatter(results["normal"]["evals"], results["normal"]["fitness"], label="W/o archive",
				color="red", facecolors='none', alpha=1)
	plt.scatter(results["archive"]["evals"], results["archive"]["fitness"], label="W archive",
				color="blue", facecolors='none', alpha=1)
	plt.legend()
	plt.yscale('log')
	plt.title("Convergence graph for MOEDA (L=20, pop=100)")
	plt.xlabel("Evaluations")
	plt.ylabel("Hypervolume")
	plt.figure(1)
	plt.scatter(results["normal"]["xs"], results["normal"]["ys"], label="W/o archive", color="red", facecolors='none', alpha=0.3)
	plt.scatter(results["archive"]["xs"], results["archive"]["ys"], label="W archive", color="blue", facecolors='none', alpha=0.3)
	plt.xlabel("f0")
	plt.ylabel("f1")
	plt.legend()
	plt.show()


def q2b(run=False, name="run.json"):
	result = {"normal": {L: {"means": [], "sizes": []} for L in PROBLEMS},
			  "archive": {L: {"means": [], "sizes": []} for L in PROBLEMS}}
	variants = ["normal", "archive"]
	precision = 0.00001
	for problem in PROBLEMS:
		for ea in variants:
			plt.figure()
			plt.title(ea+": L="+str(problem))
			plt.yscale('log')
			last_val = -1.0
			res = 0.0
			size = 4
			while last_val*precision <= abs(res-last_val) and size != 10**4:
				last_val = res
				size = min(size*2, 10**4)
				fitness, evals, _, _ = run_reliably(5, problem, size, ea=="normal")
				result[ea][problem]["sizes"].append(size)
				res = np.mean(fitness)
				result[ea][problem]["means"].append(res)
				with open("results/b/"+name, "w") as f:
					json.dump(result, f, indent=4)
			ub_v = result[ea][problem]["means"][-1]
			lb_v = result[ea][problem]["means"][-2]
			if ub_v == lb_v:
				lb_v = result[ea][problem]["means"][-3]
				ub = result[ea][problem]["sizes"][-2]
				lb = result[ea][problem]["sizes"][-3]
			else:
				ub = result[ea][problem]["sizes"][-1]
				lb = result[ea][problem]["sizes"][-2]
			while lb_v*precision >= abs(ub_v-lb_v):
				if ub == lb:
					lb = int(lb*3/4)
				size = int((ub+lb)/2)
				fitness, evals, _, _ = run_reliably(5, problem, size, ea == "normal")
				res = np.mean(fitness)
				result[ea][problem]["sizes"].append(size)
				result[ea][problem]["means"].append(res)
				ub = size
				ub_v = res
				with open("results/b/"+name, "w") as f:
					json.dump(result, f, indent=4)
			plt.scatter(result[ea][problem]["sizes"], result[ea][problem]["means"])
	plt.show()


#q2a(False, "run02.json")
q2b(True, "run02.json")
#run_moeda(20, 100, True)
