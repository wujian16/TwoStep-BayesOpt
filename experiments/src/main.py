import numpy as np
np.random.seed(12345)
import random
random.seed(12345)
import pandas as pd
import os, sys
import time

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean
from moe.optimal_learning.python.cpp_wrappers.robust_knowledge_gradient_mcmc import LowerConfidenceBoundMCMC
from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import ProbabilityImprovementMCMC
from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import ExpectedImprovementMCMC
from moe.optimal_learning.python.cpp_wrappers.two_step_expected_improvement_mcmc import TwoStepExpectedImprovementMCMC

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.optimal_learning.python.default_priors import DefaultPrior

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

import bayesian_optimization
import synthetic_functions
import hyper_tuning_functions
import ATO_wrapper

import synthetic_functions_remi

# arguments for calling this script:
# python main.py [obj_func_name] [method_name] [num_to_sample] [job_id]
# example: python main.py Branin TS 1 1
# you can define your own obj_function and then just change the objective_func object below, and run this script.

argv = sys.argv[1:]
obj_func_name = str(argv[0])
method = str(argv[1])
num_to_sample = int(argv[2])
job_id = int(argv[3])
dirs = '/fs/home/jw926/TwoStage/experiments/Raw.Results/'

# create a folder, change current directory to that folder
while True:
    directory = dirs + obj_func_name + '/'
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        break
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise
            # time.sleep might help here
        pass

while True:
    directory = dirs + obj_func_name + '/' + method + '/'
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        break
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise
            # time.sleep might help here
        pass

# constants
num_func_eval = 50

obj_func_dict = {'Branin': synthetic_functions.Branin(),
                 'Camel': synthetic_functions.Camel(),
                 'Rosenbrock': synthetic_functions.Rosenbrock(),
                 'Hartmann3': synthetic_functions.Hartmann3(),
                 'Levy10': synthetic_functions.Levy(),
                 'Hartmann6': synthetic_functions.Hartmann6(),
                 'Ackley': synthetic_functions.Ackley(),
                 'Cosine': synthetic_functions.Cosine(),
                 'svm_on_grid': hyper_tuning_functions.hyper_tuning_grid('svm_on_grid'),
                 'lda_on_grid': hyper_tuning_functions.hyper_tuning_grid('lda_on_grid'),
                 'logreg_on_grid': hyper_tuning_functions.hyper_tuning_grid('logreg_on_grid'),
                 'ato': ATO_wrapper.ATO(),
                 'Branin_remi': synthetic_functions_remi.Branin(),
                 'Camel_remi': synthetic_functions_remi.Camel(),
                 'Goldstein_remi': synthetic_functions_remi.Goldstein(),
                 'Griewank_remi': synthetic_functions_remi.Griewank()}

if 'remi' in obj_func_name:
    num_func_eval = 14

num_iteration = int(num_func_eval / num_to_sample) + 1

objective_func = obj_func_dict[obj_func_name]
dim = int(objective_func._dim)
num_initial_points = int(objective_func._num_init_pts)

num_fidelity = objective_func._num_fidelity

inner_search_domain = pythonTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                                 for i in xrange(objective_func._search_domain.shape[0]-num_fidelity)])
cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
cpp_inner_search_domain = cppTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
                                                  for i in xrange(objective_func._search_domain.shape[0]-num_fidelity)])

# get the initial data
init_pts = np.zeros((objective_func._num_init_pts, objective_func._dim))
init_pts[:, :objective_func._dim-objective_func._num_fidelity] = inner_search_domain.generate_uniform_random_points_in_domain(objective_func._num_init_pts)
for pt in init_pts:
    pt[objective_func._dim-objective_func._num_fidelity:] = np.ones(objective_func._num_fidelity)

# observe
derivatives = objective_func._observations
observations = [0] + [i+1 for i in derivatives]
init_pts_value = np.array([objective_func.evaluate(pt) for pt in init_pts])#[:, observations]
true_value_init = np.array([objective_func.evaluate_true(pt) for pt in init_pts])#[:, observations]

init_data = HistoricalData(dim = objective_func._dim, num_derivatives = len(derivatives))
init_data.append_sample_points([SamplePoint(pt, [init_pts_value[num, i] for i in observations],
                                            objective_func._sample_var) for num, pt in enumerate(init_pts)])

# initialize the model
prior = DefaultPrior(1+dim+len(observations), len(observations))

# noisy = False means the underlying function being optimized is noise-free
cpp_gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data = init_data,
                                                           derivatives = derivatives,
                                                           prior = prior,
                                                           chain_length = 1000,
                                                           burnin_steps = 2000,
                                                           n_hypers = 2 ** 4,
                                                           noisy = True)
cpp_gp_loglikelihood.train()

py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=100,
                                               max_num_restarts=3,
                                               num_steps_averaged=15,
                                               gamma=0.7,
                                               pre_mult=0.01,
                                               max_relative_change=0.02,
                                               tolerance=1.0e-8)

py_sgd_params_acquisition = pyGradientDescentParameters(max_num_steps=50,
                                               max_num_restarts=1,
                                               num_steps_averaged=0,
                                               gamma=0.7,
                                               pre_mult=1.0,
                                               max_relative_change=0.1,
                                               tolerance=1.0e-8)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1,
                                                 max_num_steps=6,
                                                 max_num_restarts=1,
                                                 num_steps_averaged=3,
                                                 gamma=0.0,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.1,
                                                 tolerance=1.0e-8)

cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=int(1000),
                                                 max_num_steps=60,
                                                 max_num_restarts=3,
                                                 num_steps_averaged=0,
                                                 gamma=0.7,
                                                 pre_mult=1.0,
                                                 max_relative_change=0.1,
                                                 tolerance=1.0e-8)

# minimum of the mean surface
cpp_gp = cpp_gp_loglikelihood.models[0]
report_point = (cpp_gp.get_historical_data_copy()).points_sampled[np.argmin(cpp_gp._points_sampled_value[:, 0])]
print cpp_gp._covariance._hyperparameters

# the results we need to record.
points_sampled = pd.DataFrame(np.zeros((num_iteration*num_to_sample, dim+len(observations))))
points_reported = pd.DataFrame(objective_func._min_value * np.ones((num_iteration+1, dim+len(observations)+1)))
time_iterations = pd.DataFrame(np.zeros((4*num_iteration, 1)))

capital_so_far = objective_func._num_init_pts
# the true value so far in the initial data
points_reported.iloc[0, :dim] = report_point
points_reported.iloc[0, dim:-1] = objective_func.evaluate_true(report_point)[observations]
points_reported.iloc[0, -1] = capital_so_far
print "best so far in the initial data {0}".format(points_reported.iloc[0, dim])
points_sampled.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".points.sampled.csv", sep=",")
points_reported.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".points.reported.csv", sep=",")
time_iterations.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".time.csv", sep=",")

for n in xrange(num_iteration):
    print method + ", {0}th job, {1}th iteration, func={2}, q={3}".format(
            job_id, n, obj_func_name, num_to_sample
    )
    time1 = time.time()

    if method == 'KG' or \
       method == 'rKG' or \
       method == 'cycling-two-step' or \
       method == 'TS' or \
       method == 'two-step+':
        discrete_pts_list = []
        discrete, _ = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_search_domain,
                                                                     cpp_sgd_params_kg, 2, num_mc=2 ** 12)
        for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
            discrete_pts_optima = np.array(discrete)

            eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gp.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                                  (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim-objective_func._num_fidelity))

            test = np.zeros(eval_pts.shape[0])
            ps_evaluator = PosteriorMean(cpp_gp, num_fidelity)
            for i, pt in enumerate(eval_pts):
                ps_evaluator.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
                test[i] = -ps_evaluator.compute_objective_function()

            initial_point = eval_pts[np.argmin(test)]

            ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
            report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)

            ps_evaluator.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            if -ps_evaluator.compute_objective_function() > np.min(test):
                report_point = initial_point

            discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                             (discrete_pts_optima.shape[0] + 1, cpp_gp.dim-objective_func._num_fidelity))
            discrete_pts_list.append(discrete_pts_optima)

        ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0], num_fidelity)
        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
        if method == "KG":
            # KG method
            next_points, voi = bayesian_optimization.gen_sample_from_qkg_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                                                              ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
                                                                              cpp_sgd_params_kg, num_to_sample, num_mc=2 ** 8)
        elif method == 'rKG':
            # robust KG method
            next_points, voi = bayesian_optimization.gen_sample_from_rKG_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                                                              ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
                                                                              cpp_sgd_params_kg, num_to_sample, 1.0, num_mc=2 ** 6)
        else:
            # two-step method
            if method == 'cycling-two-step':
                left = num_iteration-n
                # check odd or even
                remaining = left % 2
                factor = 1 - remaining
            elif method == 'TS':
                factor = 1.0#min(num_iteration-n-1, 1)
                # factor = 0.0
                # if num_iteration-n-1 > 0:
                #     factor = 0.5*(1-pow(0.5, num_iteration-n-1))/(1-0.5)
            else:
                factor = min(num_iteration-n-1, 2)
            next_points, voi = bayesian_optimization.gen_sample_from_two_step_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                                                                   ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
                                                                                   cpp_sgd_params_kg, num_to_sample, factor,
                                                                                   num_mc=2 ** 6)
    elif method == 'EI':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e4))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        # EI method
        next_points, voi = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_search_domain,
                                                                          cpp_sgd_params_kg, eval_pts, num_to_sample, num_mc=2 ** 12)
    elif method == 'GP-LCB':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e4))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        pvar = LowerConfidenceBoundMCMC(cpp_gp_loglikelihood.models, num_fidelity)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            pvar.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -pvar.compute_objective_function()

        initial_points = np.zeros((20, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        indices = np.argsort(test)
        for i in range(20):
            initial_points[i, :] = eval_pts[indices[i]]

        #initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        pvar_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, pvar, py_sgd_params_acquisition)
        report_point = multistart_optimize(pvar_mean_opt, initial_points, num_multistarts = 20)[0]

        pvar.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -pvar.compute_objective_function() > np.min(test):
            report_point = initial_points[[0]]
        next_points = report_point
        voi = np.nan
    elif method == 'PI':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        pvar = ProbabilityImprovementMCMC(cpp_gp_loglikelihood._gaussian_process_mcmc, num_to_sample)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            pvar.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -pvar.compute_objective_function()

        initial_points = np.zeros((10, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        indices = np.argsort(test)
        for i in range(10):
            initial_points[i, :] = eval_pts[indices[i]]

        #initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        pvar_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, pvar, py_sgd_params_acquisition)
        report_point = multistart_optimize(pvar_mean_opt, initial_points, num_multistarts = 10)[0]

        pvar.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -pvar.compute_objective_function() > np.min(test):
            report_point = initial_points[[0]]
        next_points = report_point
        voi = np.nan
    elif method == 'MEI':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(100))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        pvar = ExpectedImprovementMCMC(cpp_gp_loglikelihood._gaussian_process_mcmc, num_to_sample)
        test = np.zeros(eval_pts.shape[0])

        for i, pt in enumerate(eval_pts):
            print eval_pts[i]
            pvar.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -pvar.compute_objective_function()
            print -test[i]

        initial_points = np.zeros((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        indices = np.argsort(test)
        # noinspection PyUnboundLocalVariable
        for i in range(1):
            initial_points[i, :] = eval_pts[indices[i]]
        print initial_points

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        pvar_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, pvar, py_sgd_params_acquisition)
        report_point = multistart_optimize(pvar_mean_opt, initial_points, num_multistarts = 1)[0]

        pvar.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -pvar.compute_objective_function() > np.min(test):
            report_point = initial_points[[0]]
        print report_point
        next_points = report_point
        pvar.set_current_point(next_points.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        voi = pvar.compute_objective_function()
    elif method == 'MTS':
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e2))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))


        discrete_pts_list = []
        discrete =  inner_search_domain.generate_uniform_random_points_in_domain(int(10))
        for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
            discrete_pts_optima = np.array(discrete)

            eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
            eval_pts = np.reshape(np.append(eval_pts,
                                            (cpp_gp.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                                  (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim-objective_func._num_fidelity))

            test = np.zeros(eval_pts.shape[0])
            ps_evaluator = PosteriorMean(cpp_gp, num_fidelity)
            for i, pt in enumerate(eval_pts):
                ps_evaluator.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
                test[i] = -ps_evaluator.compute_objective_function()

            initial_point = eval_pts[np.argmin(test)]

            ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
            report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)

            ps_evaluator.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            if -ps_evaluator.compute_objective_function() > np.min(test):
                report_point = initial_point

            discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                             (discrete_pts_optima.shape[0] + 1, cpp_gp.dim-objective_func._num_fidelity))
            discrete_pts_list.append(discrete_pts_optima)

        ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0], num_fidelity)
        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)

        pvar = TwoStepExpectedImprovementMCMC(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                              num_fidelity,
                                              ps_sgd_optimizer,
                                              discrete_pts_list,
                                              num_to_sample,
                                              1.0,
                                              num_mc_iterations=2 ** 6)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            pvar.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -pvar.compute_objective_function()

        initial_points = np.zeros((10, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        indices = np.argsort(test)
        for i in range(10):
            initial_points[i, :] = eval_pts[indices[i]]

        #initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        pvar_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, pvar, py_sgd_params_acquisition)
        report_point = multistart_optimize(pvar_mean_opt, initial_points, num_multistarts = 10)[0]

        pvar.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -pvar.compute_objective_function() > np.min(test):
            report_point = initial_points[[0]]
        next_points = report_point
        voi = np.nan
    else:
        print method + str(" not supported")
        sys.exit(0)

    print method + " takes "+str((time.time()-time1))+" seconds"
    time_iterations.iloc[4*n, 0] = (time.time()-time1)/60
    time1 = time.time()
    print method + " suggests points:"
    print next_points

    # evaluate the points
    for num, pt in enumerate(next_points):
        points_sampled.iloc[n*num_to_sample+num, :dim] = pt

    time_iterations.iloc[4*n+1, 0] = (time.time()-time1)/60

    points_sampled.iloc[n*num_to_sample: (n+1)*num_to_sample, dim:] = np.array([objective_func.evaluate(pt) for pt in next_points])[:, observations]
    sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[observations], objective_func._sample_var) for pt in next_points]

    #print "evaluating takes "+str((time.time()-time1)/60)+" mins"
    capitals = np.ones(num_to_sample)
    for i in xrange(num_to_sample):
        if num_fidelity > 0:
            value = 1.0
            for j in xrange(num_fidelity):
                value *= next_points[i, dim-1-j]
            capitals[i] = value
    capital_so_far += np.amax(capitals)
    print "evaluating takes capital " + str(capital_so_far) +" so far"

    # retrain the model
    time1 = time.time()
    # if method == 'twostep':
    #     cpp_gp_copy = cppGaussianProcessLogLikelihoodMCMC(historical_data = cpp_gp_loglikelihood.get_historical_data_copy(),
    #                                                       derivatives = derivatives,
    #                                                       prior = prior,
    #                                                       chain_length = 1000,
    #                                                       burnin_steps = 2000,
    #                                                       n_hypers = 2 ** 4,
    #                                                       burned = True,
    #                                                       starting_pos = np.array(cpp_gp_loglikelihood.p0),
    #                                                       noisy = False)
    #     cpp_gp_copy.train()

    cpp_gp_loglikelihood.add_sampled_points(sampled_points)
    cpp_gp_loglikelihood.train()

    time_iterations.iloc[4*n+2, 0] = (time.time()-time1)/60

    print "retraining the model takes "+str((time.time()-time1))+" seconds"
    time1 = time.time()

    # report the point
    if method == "EI" or method == "GP-LCB" or method == "PI" or method == "MEI" or method == "MTS":
        cpp_gp = cpp_gp_loglikelihood.models[0]
        report_point = (cpp_gp.get_historical_data_copy()).points_sampled[np.argmin(cpp_gp._points_sampled_value[:, 0])]
        #print cpp_gp._points_sampled_value
    # elif method == 'twostep': or 'two-step' in method
    #     next_points, voi = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_copy._gaussian_process_mcmc, cpp_search_domain,
    #                                                                       cpp_sgd_params_kg, num_to_sample, num_mc=2 ** 10)
    #     print method + " suggests points if at the final iteration:"
    #     print next_points
    #     sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[observations], objective_func._sample_var) for pt in next_points]
    #     cpp_gp_copy.add_sampled_points(sampled_points)
    #     cpp_gp_copy.train()
    #
    #     cpp_gp = cpp_gp_copy.models[0]
    #     report_point = (cpp_gp.get_historical_data_copy()).points_sampled[np.argmin(cpp_gp._points_sampled_value[:, 0])]
    else:
        eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e5))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
                              (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
        test = np.zeros(eval_pts.shape[0])
        for i, pt in enumerate(eval_pts):
            ps.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
            test[i] = -ps.compute_objective_function()

        initial_points = np.zeros((10, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))
        indices = np.argsort(test)
        for i in range(10):
            initial_points[i, :] = eval_pts[indices[i]]

        #initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

        py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
        ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
        report_point = multistart_optimize(ps_mean_opt, initial_points, num_multistarts = 10)[0]

        ps.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
        if -ps.compute_objective_function() > np.min(test):
            report_point = initial_points[0]

    report_point = report_point.ravel()
    report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

    points_reported.iloc[n+1,:dim] = report_point

    print "recommended points: ",
    print report_point
    result = objective_func.evaluate_true(report_point)
    points_reported.iloc[n+1, dim:-1] = np.array([result[i] for i in observations])
    points_reported.iloc[n+1, dim+len(observations)] = capital_so_far

    time_iterations.iloc[4*n+3, 0] = (time.time()-time1)/60
    print method+", VOI {0}, best so far {1}".format(voi, points_reported.iloc[n+1,dim])

    points_sampled.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".points.sampled.csv", sep=",")
    points_reported.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".points.reported.csv", sep=",")
    time_iterations.to_csv(dirs + obj_func_name + "/" + method + "/" + str(job_id) + ".time.csv", sep=",")