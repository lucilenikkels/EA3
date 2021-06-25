class Run_NSGAII:
    def __init__(self, L_list, archive=False, populationSize=100, addon="", loops=5
                 ):
        self.archive = archive
        self.L_list = L_list
        self.L = L_list[0]  # number of (discrete) variables
        self.addon = addon
        self.file_name = self.file_name_creator()
        self.populationSize = populationSize
        self.loops = loops
        self.results = {L: {} for L in self.L_list}
        self.evals = 10**4
        self.max_popSize = self.evals-1
        self.pop_size_factor = 4

    def file_name_creator(self):
        if self.archive:
            self.addon += "_archive"
        filename = "Results/NSGAII_" + self.addon + ".json"
        return filename

    def setup_results(self):
        self.results[self.L][self.populationSize] = {'raw': {}, 'mean': {}, 'std': {}, 'var': {}}

    def setup_NSGA(self):
        EA = NSGAII(populationSize=self.populationSize,
            numberOfVariables=self.L,
            numberOfEvaluations=self.evals,
            fitnessFunction='knapsack',
            selection=variation.selection, crossover=variation.crossover, mutation=variation.mutation,
            hillClimber=None,
            tournamentSize=2, crossoverProb=0.9, mutationProb='auto',
            randomSeed=42, archive=self.archive)
        return EA

    def run_EA(self):
        self.setup_results()
        for run in range(0, self.loops):
            EA = self.setup_NSGA()
            EA.evolve()  # Run algorithm
            self.store_results(EA)
        self.parse_results()

    def run_EA_and_save(self):
        self.run_EA()
        self.save_results()

    def store_results(self, EA):
        for j, eval in enumerate(EA.numberOfEvaluationsByGeneration):
            try:
                self.results[self.L][self.populationSize]['raw'][eval].append(EA.hyperVolumeByGeneration[j])
            except KeyError:
                self.results[self.L][self.populationSize]['raw'][eval] = [EA.hyperVolumeByGeneration[j]]
        self.results[self.L][self.populationSize]['evals'] = EA.numberOfEvaluationsByGeneration

    def parse_results(self):
        results_tmp = self.results[self.L][self.populationSize]['raw']
        for key in results_tmp:
            self.results[self.L][self.populationSize]['mean'][key] = np.mean(results_tmp[key])
            self.results[self.L][self.populationSize]['std'][key] = np.std(results_tmp[key])
            self.results[self.L][self.populationSize]['var'][key] = np.var(results_tmp[key])

            try:
                self.results[self.L][self.populationSize]['means'].append(np.mean(results_tmp[key]))
            except KeyError:
                self.results[self.L][self.populationSize]['means'] = [np.mean(results_tmp[key])]
            try:
                self.results[self.L][self.populationSize]['stds'].append(np.std(results_tmp[key]))
            except KeyError:
                self.results[self.L][self.populationSize]['stds'] = [np.std(results_tmp[key])]
            try:
                self.results[self.L][self.populationSize]['vars'].append(np.var(results_tmp[key]))
            except KeyError:
                self.results[self.L][self.populationSize]['vars'] = [np.var(results_tmp[key])]

    def save_results(self):
        with open(self.file_name, 'w') as file:
            NSGA_results = json.dumps(self.results, indent=4)
            file.write(NSGA_results)
            file.close()