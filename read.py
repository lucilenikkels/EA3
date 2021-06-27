import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import json


def get_coords(data):
	filenames = os.listdir(f'{data}')

	results = {}

	for filename in filenames:
		info = filename[:-4].split('_')
		problem_size = int(info[0])
		population_size = int(info[1])
		useElitistArchive = info[2]
		with open(f'{data}/{filename}') as f_out:
			lines = f_out.readlines()
			total_runs = len(lines) // 2
			x_coords = np.zeros(len(lines[0].split()))
			y_coords = np.zeros(len(lines[0].split()))
			variance_x_coords = []
			variance_y_coords = []
			
			for index in range(0, len(lines), 2):
				y_coords += np.fromstring(lines[index], dtype=float, sep=' ')
				x_coords += np.fromstring(lines[index + 1], dtype=float, sep=' ')
				variance_y_coords = np.append(variance_y_coords, np.fromstring(lines[index], dtype=float, sep=' '))
				variance_x_coords = np.append(variance_x_coords, np.fromstring(lines[index + 1], dtype=float, sep=' '))
			f_out.close()
		y_coords = y_coords / float(total_runs)
		x_coords = x_coords / float(total_runs)
		zipped_coords = zip(x_coords, y_coords)
		sorted_pair_coords = sorted(zipped_coords)
		split_pair = zip(*sorted_pair_coords)
		x_coords, y_coords = [list(tuple) for tuple in split_pair]
		results[f'{problem_size}_{useElitistArchive}'] = {'x_coords': x_coords,# .tolist(),
														  'y_coords': y_coords, #.tolist(),
														  'variance_x_coords': variance_x_coords,
														  'variance_y_coords': variance_y_coords,
														  'population_size': population_size}
	return results


moels = get_coords('data')

with open("NSGAII.json", "r") as f:
	nsga = json.load(f)

with open("results/group/moeda_final.json", "r") as f:
	moeda = json.load(f)


fig, axs = plt.subplots(1, 3)
sizes = [15, 30, 60]
moeda_sizes = [1987, 4610, 1188]
nsga_sizes = [351, 470, 707]

for idx, size in enumerate(sizes):
	ax = axs[idx]
	ax.set_title("L="+str(size), fontsize=24)
	ax.set_xlabel("Evaluations", fontsize=20)
	ax.set_ylabel("Hypervolume", fontsize=20)
	ax.set_yscale('log')
	#ax.set_xscale('log')
	ax.plot(moels[str(size)+"_True"]["x_coords"], moels[str(size)+"_True"]["y_coords"], label="MOELS (pop="+str(moels[str(size)+"_True"]["population_size"])+")")
	ax.scatter(moels[str(size)+"_True"]["variance_x_coords"], moels[str(size)+"_True"]["variance_y_coords"], alpha=0.3)

	if size == 60:
		ax.plot(moeda[str(size)]["xs"], moeda[str(size)]["means"], label="MOEDA (pop="+str(moeda_sizes[idx])+")")
	else:
		ax.plot(moeda[str(size)]["xs"], moeda[str(size)]["means"], label="MOEDA (pop=" + str(moeda_sizes[idx]) + ")")
	upper = []
	lower = []
	for y in range(len(moeda[str(size)]["means"])):
		upper.append(moeda[str(size)]["means"][y] + moeda[str(size)]["stds"][y])
		lower.append(moeda[str(size)]["means"][y] - moeda[str(size)]["stds"][y])
	ax.fill_between(moeda[str(size)]["xs"], upper, lower, alpha=0.3)

	ax.plot(nsga[str(size)][str(nsga_sizes[idx])]["evals"], nsga[str(size)][str(nsga_sizes[idx])]["means"], label="NSGAII (pop=" + str(nsga_sizes[idx]) + ")")
	upper = []
	lower = []
	for y in range(len(nsga[str(size)][str(nsga_sizes[idx])]["means"])):
		upper.append(nsga[str(size)][str(nsga_sizes[idx])]["means"][y] + nsga[str(size)][str(nsga_sizes[idx])]["stds"][y])
		lower.append(nsga[str(size)][str(nsga_sizes[idx])]["means"][y] - nsga[str(size)][str(nsga_sizes[idx])]["stds"][y])
	ax.fill_between(nsga[str(size)][str(nsga_sizes[idx])]["evals"], upper, lower, alpha=0.3)

	ax.legend(fontsize=18)
plt.show()


#data.get('problemsize_True').get('x_coords')
#data.get('problemsize_True').get('y_coords')
#data.get('problemsize_True').get('variance_x_coords')
#data.get('problemsize_True').get('variance_y_coords')

