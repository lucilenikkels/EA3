class Analysis:
    def __init__(self, filenames):
        self.filenames = filenames
        self.results = {}
        for filename in self.filenames:
            self.results[filename] = self.read_json_file(filename)
        self.pops, self.L_list = self.determine_pop()

    def read_json_file(self, filename):
        with open(filename, 'r') as f_:
            results = json.load(f_)
        return results

    def read_text_file(self, filename):
        with open(filename, 'r') as f_:
            tmp_str = f_.readlines()
            tmp_str = [j.strip('\n') for j in tmp_str]
            result = []
            for j in tmp_str:
                result.append(float(j))
        return result

    def determine_pop(self):
        pops = {}
        for filename in self.filenames:
            pops[filename] = []
            L_list = []
            for L in self.results[filename]:
                pops[filename].append(list(self.results[filename][L].keys())[0])
                pops[filename] = [int(x) for x in pops[filename]]
                L_list.append(int(L))
        return pops, L_list

    def plot_convergence(self):
        plt.style.use('./src/presentation.mplstyle')
        # Setting up the plot
        nrows = 1
        # print('Number of rows:', nrows)
        ncols = len(self.L_list)
        if ncols == 1:
            plt.style.use('./src/presentation.mplstyle')
        elif ncols == 3:
            plt.style.use('./src/presentation2.mplstyle')
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

        # Choosing the subplots
        ncol = 0
        for k, L in enumerate(self.L_list):
            L = str(L)
            try:
                axs_tmp = axs[ncol]
            except TypeError:
                axs_tmp = axs
            # Setting up text
            if ncol == 0:
                axs_tmp.set_ylabel("Fitness")
            axs_tmp.set_xlabel("Fitness evaluations (l=" + str(L) + ')')
            axs_tmp.set_yscale('log')

            # Filling in the subplot
            for filename in self.filenames:
                pop = str(self.pops[filename][k])
                axs_tmp.plot(self.results[filename][L][pop]['evals'], self.results[filename][L][pop]['means'],
                             label='W/out archive, pop=' + pop)
                axs_tmp.fill_between(self.results[filename][L][pop]['evals'],
                                     [self.results[filename][L][pop]['means'][j] - self.results[filename][L][pop]['stds'][j]
                                      for j in range(0, len(self.results[filename][L][pop]['means']))],
                                     [self.results[filename][L][pop]['means'][j] + self.results[filename][L][pop]['stds'][j]
                                      for j in range(0, len(self.results[filename][L][pop]['means']))],
                                     alpha=0.3)

            axs_tmp.legend(loc='best', fancybox=True)

            ncol += 1

        plt.show()