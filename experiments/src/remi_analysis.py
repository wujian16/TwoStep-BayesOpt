import os, sys
import numpy as np
import pandas as pd

import synthetic_functions_remi

# example: python remi_analysis.py Branin_remi 14 log 40 50

argv = sys.argv[1:]
obj_func_name = argv[0]
num_iterations = int(argv[1])
scale = argv[2]
jobs = int(argv[3])
percentile = int(argv[4])

path = "../Raw.Results/" + obj_func_name+"/"
folders = os.listdir(path)

#potential = ['EI', 'KG', 'GP-LCB', 'PI', 'TS']
potential = ['TS']
folders_copy = list(folders)

obj_func_dict = {'Camel_remi': synthetic_functions_remi.Camel(),
                 'Branin_remi': synthetic_functions_remi.Branin()}

evaluations = np.ones((200, len(potential)))

for num, method in enumerate(potential):
    print method
    cur_path = path+method
    temp = obj_func_dict[obj_func_name]._min_value*np.ones((200, jobs))
    index = 0
    for i in xrange(jobs):
        file = cur_path+"/"+str(i+1)+".points.reported.csv"
        if not os.path.isfile(file):
            continue
        data = pd.read_csv(file, sep=",", index_col=0)
        evaluations[:min(num_iterations+1, data.shape[0]), num] = data.iloc[:min(num_iterations+1, data.shape[0]), -1]

        temp[:min(num_iterations+1, data.shape[0]), index] = 1-(data.iloc[:min(num_iterations+1, data.shape[0]), -2].as_matrix()-obj_func_dict[obj_func_name]._min_value)/float(data.iloc[0, -2]-obj_func_dict[obj_func_name]._min_value)
        index += 1

    mean = np.percentile(temp[:min(num_iterations+1, data.shape[0]), :index], percentile, axis=1)
    #mean = np.mean(temp[:min(num_iterations+1, data.shape[0]), :index], axis=1)
    std = temp[:min(num_iterations+1, data.shape[0]), :(index+1)].std(axis=1)/np.sqrt(jobs)
    print mean
