import numpy
import scipy
from scipy.stats import ranksums, ttest_ind, kruskal


moeda = {
    15: [0.4350451963146951,
         0.43498429656438403,
         0.4306290944207129,
         0.4346745778342309,
         0.43470067772722126],
    30: [0.4096120268462381,
         0.41623271958654223,
         0.41224949312651116,
         0.4117979551714608,
         0.4012280527623732],
    60: [0.37709541174309213,
         0.37479966226264305,
         0.3727451873131413,
         0.3835163213880123,
         0.3720426250105133]
}


moels = {
    15: [0.37386704714510677,
         0.37239849316617796,
         0.36806417093689914,
         0.3746100240989012,
         0.3659170197402191],
    30: [0.39111295019242803,
         0.3737681145238186,
         0.3842629935987856,
         0.36870101057829,
         0.3761949562884233],
    60: [0.3736879669554913,
         0.3566352375461698,
         0.3595234839760351,
         0.35763075928413585,
         0.3568325799083862]
}


nsga = {
    15: [0.4353148952089297,
                    0.43529749528026934,
                    0.43488163698528837,
                    0.43506781622195345,
                    0.43498081657865206],
    30: [0.44259247091913834,
                    0.4412956390007635,
                    0.45729844235714906,
                    0.45560272861264145,
                    0.4554997462720161],
    60: [0.39795062306152224,
           0.40629303649283166,
           0.4031978155140231,
           0.39961605163979746,
           0.39548196408275044]}

sizes = [15, 30, 60]
for size in sizes:
    eda_els = kruskal(moeda[size], moels[size])
    eda_nsga = kruskal(moeda[size], nsga[size])
    nsga_els = kruskal(nsga[size], moels[size])
    print('MOEDA+MOELS (%s): %f (p.=%f)' % (size, eda_els[0], eda_els[1]))
    print('MOEDA+NSGA (%s): %f (p.=%f)' % (size, eda_nsga[0], eda_nsga[1]))
    print('NSGA+MOELS (%s): %f (p.=%f)' % (size, nsga_els[0], nsga_els[1]))
