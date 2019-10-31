import numpy
numpy.random.seed(12345)
import random
random.seed(12345)
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

from matplotlib.backends.backend_pdf import PdfPages
import pickle
import sys
import matplotlib.pyplot as plt

from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential as cppSquareExponential
from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess as cppGaussianProcess
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement as cppExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import KnowledgeGradient as cppKnowledgeGradient
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

import synthetic_functions

# arguments for calling this script:
# python speedup.py [obj_func_name] [num_to_sample] [num_lhc] [use_gpu] [which_gpu] [start_idx] [end_idx] [method_name]
# example: python EI.py GP 1 1000 0 1
argv = sys.argv[1:]
obj_func_name = argv[0]
num_to_sample = int(argv[1])
num_func_eval_dict = {"Branin": 60, "LG": 60, "Hartmann": 60, "Ackley": 100, "Rosenbrock": 100, "Levy": 60, "GP": 10, "GP_wavy" : 10}
num_iteration = int(num_func_eval_dict[obj_func_name] / num_to_sample) + 1
lhc_search_itr = int(argv[2])

start_idx = int(argv[3])
end_idx = int(argv[4])
figwidth = int(argv[5])
figheight = int(argv[6])

# constants
a=numpy.random.normal(0,1)
theta=10*numpy.random.multivariate_normal(numpy.zeros(6), numpy.identity(6))
obj_func_dict = {'Branin': synthetic_functions.Branin(),
                 'Hartmann': synthetic_functions.Hartmann3(), 'Rosenbrock': synthetic_functions.Rosenbrock(),
                 'Ackley': synthetic_functions.Ackley(), 'Levy': synthetic_functions.Levy()}
cpp_sgd_params = cppGradientDescentParameters(num_multistarts=2000, max_num_steps=20, max_num_restarts=1,
                                              num_steps_averaged=15, gamma=0.7, pre_mult=1.0,
                                              max_relative_change=0.7, tolerance=1.0e-3)

if obj_func_name == "GP":
    gp_grad_info_dict = pickle.load(open('random_gp_grad_1d', 'rb'))
    hist_data_grad = HistoricalData(gp_grad_info_dict['dim'], 1)
    hist_data_grad.append_historical_data(gp_grad_info_dict['points'], gp_grad_info_dict['values'], gp_grad_info_dict['vars'])
    objective_func = synthetic_functions.RandomGP(gp_grad_info_dict['dim'], gp_grad_info_dict['hyper_params'], hist_data_grad)
    hyper_params = gp_grad_info_dict['hyper_params']
    init_pts = [[-1.5], [-1.0], [1.0], [1.5]]
    ymax = 2
elif obj_func_name == "GP_wavy":
    gp_grad_info_dict = pickle.load(open('random_gp_1d_wavy', 'rb'))
    hist_data_grad = HistoricalData(gp_grad_info_dict['dim'], 0)
    hist_data_grad.append_historical_data(gp_grad_info_dict['points'], gp_grad_info_dict['values'], gp_grad_info_dict['vars'])
    objective_func = synthetic_functions.RandomGP(gp_grad_info_dict['dim'], gp_grad_info_dict['hyper_params'], hist_data_grad)
    hyper_params = gp_grad_info_dict['hyper_params']
    hyper_params[1] = 0.2
    init_pts =  [[-2.0], [0.0], [0.3], [0.5], [2.0]]
    ymax = 1
else:
    objective_func = obj_func_dict[obj_func_name]


#init_data = utils.get_init_data_from_db(objective_func._dim, objective_func._sample_var, utils.sql_engine, 'init_points_'+obj_func_name)
python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])

result=numpy.zeros((num_iteration, 6))
best_so_far_kg = numpy.zeros((end_idx-start_idx, num_iteration + 1))

# begin job
for job_no in xrange(start_idx, end_idx):
    python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])

    init_value = [objective_func.evaluate(pt) for pt in init_pts]

    init_data = HistoricalData(objective_func._dim, 1)
    init_data.append_sample_points([SamplePoint(pt, init_value[num], objective_func._sample_var) for num, pt in enumerate(init_pts)])

    #best_so_far_kg[job_no-start_idx, 0] = objective_func.evaluate_true(init_data.points_sampled[numpy.argmin(init_data.points_sampled_value)])
    print "best so far {0}".format(best_so_far_kg[job_no-start_idx,0])

    init_data_nogradient = HistoricalData(objective_func._dim, 0)
    init_data_nogradient.append_sample_points([SamplePoint(pt, init_value[num][0], objective_func._sample_var) for num, pt in enumerate(init_pts)])

    cpp_cov_nograd = cppSquareExponential(hyper_params)
    cpp_gp_nogradient = cppGaussianProcess(cpp_cov_nograd, numpy.ones(1)*0.0001, init_data_nogradient, [])

    discrete_pts=numpy.array(sorted(python_search_domain.generate_uniform_random_points_in_domain(100)))#generate_grid_points_in_domain(400)
    temp_mean=numpy.zeros(100)
    temp_std=numpy.zeros(100)
    with PdfPages('../Plots/EI.pdf') as pdf:
        for n in xrange(num_iteration):
            print "EI, {0}th job, {1}th iteration, func={2}, q={3}".format(
                    job_no, n, obj_func_name, num_to_sample
            )
            fig, ax = plt.subplots(figsize=(figwidth, figheight))
            ax.set_ylim(-3, ymax)
            #ax.set_aspect('equal')

            for p in xrange(len(discrete_pts)):
                temp_mean[p]=cpp_gp_nogradient.compute_mean_of_points(discrete_pts[p])[0]
                temp_std[p]=numpy.sqrt(cpp_gp_nogradient.compute_variance_of_points(discrete_pts[p])[0,0])
            pd.DataFrame(numpy.concatenate((temp_mean.reshape(100, 1), temp_std.reshape(100, 1)), axis=1)).to_csv("../Plots/EI/post"+str(n)+".csv")

            plt.plot(discrete_pts.flatten(), temp_mean, 'k')
            plt.fill_between(discrete_pts.flatten(), temp_mean-temp_std, temp_mean+temp_std, color='b', alpha=0.2)

            x_axis=cpp_gp_nogradient.get_historical_data_copy().points_sampled
            #y_axis=[cpp_gp_nogradient.compute_mean_of_points(pt) for pt in x_axis]
            y_axis=cpp_gp_nogradient.get_historical_data_copy().points_sampled_value[:, 0]
            # plt.title('posterior without gradient', fontsize=22)
            # if n == 1:
            #     plt.title('after evaluating the point by EI', fontsize=22)
            plt.plot(x_axis, y_axis, 'bs')
            #with PdfPages('../Plots/Tutorial/post_ei'+str(n)+'.pdf') as pdf:
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            cpp_ei_evaluator_nogradient = cppExpectedImprovement(gaussian_process=cpp_gp_nogradient, num_mc_iterations=1000)
            xlist = discrete_pts.flatten()

            Y = numpy.zeros(100)
            eimax = -float("inf")
            next_point_ei = None
            for i in xrange(len(xlist)):
                cpp_ei_evaluator_nogradient.set_current_point(xlist[i])
                Y[i] = cpp_ei_evaluator_nogradient.compute_objective_function()
                if Y[i]>eimax:
                    eimax=Y[i]
                    next_point_ei=xlist[i]
            pd.DataFrame(numpy.concatenate((discrete_pts, numpy.array(Y).reshape(100, 1)), axis=1)).to_csv("../Plots/EI/ei"+str(n)+".csv")

            fig, ax = plt.subplots(figsize=(figwidth, figheight))
            ax.set_ylim(0, 0.5)
            #ax.set_aspect('equal')
            plt.plot(discrete_pts.flatten(), Y, 'g', linestyle = '-.')
            plt.plot(next_point_ei, eimax, 'rs')
            plt.legend(['EI'], fontsize=20, loc='upper left')
            #with PdfPages('../Plots/Tutorial/EI'+str(n)+'.pdf') as pdf:
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[0], objective_func._sample_var) for pt in [[next_point_ei]]]
            cpp_gp_nogradient.add_sampled_points(sampled_points)
            print cpp_gp_nogradient.get_historical_data_copy().points_sampled_value
