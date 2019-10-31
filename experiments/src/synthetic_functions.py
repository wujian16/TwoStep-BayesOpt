import numpy
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import math
#from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential as cppSquareExponential
#from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess as cppGaussianProcess

class Branin(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[0, 15], [-5, 15]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.397887
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
        is at x = [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.

            :param x[2]: 2-dim numpy array
        """
        a = 1
        b = 5.1 / (4 * pow(numpy.pi, 2.0))
        c = 5 / numpy.pi
        r = 6
        s = 10
        t = 1 / (8 * numpy.pi)
        return numpy.array([(a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1 - t) * numpy.cos(x[0]) + s),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r) * (-2* b * x[0] + c) + s * (1 - t) * (-numpy.sin(x[0]))),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r))])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Camel(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[-2., 2.], [-1., 1.]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -7.e-5#0.0
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
            :param x[2]: 2-dim numpy array
        """
        x2 = math.pow(x[0],2)
        x4 = math.pow(x[0],4)
        y2 = math.pow(x[1],2)

        return numpy.array([((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x[0]*x[1] + (-4.0 + 4.0 * y2) * y2+1.0316)/2])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Michalewicz(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[-2., 2.], [-1., 1.]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -7.e-5#0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
            :param x[2]: 2-dim numpy array
        """
        x2 = math.pow(x[0],2)
        x4 = math.pow(x[0],4)
        y2 = math.pow(x[1],2)

        return numpy.array([((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x[0]*x[1] + (-4.0 + 4.0 * y2) * y2+1.0316)/2])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Rosenbrock(object):
    def __init__(self):
        self._dim = 10
        self._search_domain = numpy.repeat([[-0.5, 0.5]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1, 1, 1)

            :param x[4]: 4-dimension numpy array
        """
        x = 40*x
        value = 0.0
        for i in range(self._dim-1):
            value += pow(1. - x[i], 2.0) + 100. * pow(x[i+1] - pow(x[i], 2.0), 2.0)
        results = [value/1.e6]
        for i in range(self._dim-1):
            results += [(2.*(x[i]-1) - 400.*x[i]*(x[i+1]-pow(x[i], 2.0)))/(1+value)]
        results += [(200. * (x[self._dim-1]-pow(x[self._dim-2], 2.0)))/(1+value)]
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)

class Hartmann3(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -3.86278
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 3
            Global minimum is -3.86278 at (0.114614, 0.555649, 0.852547)

            :param x[3]: 3-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[3., 10., 30.], [0.1, 10., 35.], [3., 10., 30.], [0.1, 10., 35.]])
        P = 1e-4 * numpy.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        results = [0.0]*4
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results[0] -= alpha[i] * numpy.exp(inner_value)
            for j in xrange(self._dim-self._num_fidelity):
                results[j+1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i,j] * (x[j] - P[i, j]))
        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        return t

class Levy(object):
    def __init__(self):
        self._dim = 10
        self._search_domain = numpy.repeat([[-1., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1, 1, 1)

            :param x[4]: 4-dimension numpy array

            a difficult test case for KG-type methods.
        """
        x = numpy.asarray_chkfinite(x)
        x = 10*x
        n = len(x)
        z = 1 + (x - 1) / 4

        results = [0] * (n+1)
        results[0] = (sin( pi * z[0] )**2
                      + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
                      +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))/40.
        results[1] = 2. * sin(pi * z[0]) * cos(pi * z[0]) * pi * (0.25)
        results[n] = (((z[-1] - 1)**2) * (2. * sin(2 * pi * z[-1]) * cos(2. * pi * z[-1]) * 2. * pi *(0.25))
                      + 2. * (z[-1]-1) * (0.25) * (1 + sin( 2. * pi * z[-1] )**2 ))

        results[1:-1] += (((z[:-1] - 1)**2) * (20. * sin(pi * z[:-1] + 1) * cos(pi * z[:-1] + 1) * pi *(0.25))
                          + 2 * (z[:-1]-1) * (0.25) * (1 + 10. * sin(pi * z[:-1] + 1)**2 ))
        return numpy.array(results)


    def evaluate(self, x):
        t = self.evaluate_true(x)
        return t

class Hartmann6(object):
    def __init__(self):
        self._dim = 6
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -3.32237
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 6
            Global minimum is -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

            :param x[6]: 6-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[10, 3, 17, 3.50, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                         [17, 8, 0.05, 10, 0.1, 14]])
        P = 1.0e-4 * numpy.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                                  [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
        results = [0.0]*7
        for i in xrange(4):
            inner_value = 0.0
            for j in xrange(self._dim-self._num_fidelity):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results[0] -= alpha[i] * numpy.exp(inner_value)
            for j in xrange(self._dim-self._num_fidelity):
                results[j+1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i,j] * (x[j] - P[i, j]))
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)

class Ackley(object):
    def __init__(self):
        self._dim = 5
        self._search_domain = numpy.repeat([[-1., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        x = 20.*x
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(x))
        results=[(-20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e)/6.]
        for i in range(int(n)):
            results += [-20.0*math.exp(-0.2*math.sqrt(firstSum/n)) * (-0.2*(x[i]/n)/(math.sqrt(firstSum/n))) -
                        math.exp(secondSum/n) * (2.0*math.pi/n) * (-math.sin(2.0*math.pi*x[i]))]

        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        results = []
        for r in t:
            n = numpy.random.normal(0, numpy.sqrt(self._sample_var))
            results += [r+n]
        return numpy.array(results)

class Cosine(object):
    def __init__(self):
        self._dim = 8
        self._search_domain = numpy.repeat([[-1., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -0.4
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (0, 0, ..., 0)

            :param x[2]: 2-dimension numpy array
        """
        value = 0.0
        for i in xrange(len(x)):
            value += -0.1*cos(5*pi*x[i]) + (x[i]**2)
        results = [value/2.]
        for i in xrange(self._dim):
            results += [0.5*pi*sin(5*pi*x[i]) + 2*x[i]]
        return numpy.array(results)

    def evaluate(self, x):
        t = self.evaluate_true(x)
        results = []
        for r in t:
            n = numpy.random.normal(0, numpy.sqrt(self._sample_var))
            results += [r+n]
        return numpy.array(results)

# class RandomGP(object):
#     def __init__(self, dim, hyper_params, historical_data=None):
#         self._cov = cppSquareExponential(hyper_params)
#         self._dim = dim
#         self._search_domain = numpy.repeat([[-2., 2.]], dim, axis=0)
#         self._hyper_domain = numpy.array([[0,100],[0,1],[0,0.01]])
#         self._hyper_params = hyper_params
#         self._num_init_pts = dim
#         self._sample_var = 1.0e-4
#         if historical_data is not None:
#             self._gp = cppGaussianProcess(self._cov, numpy.ones(1) * self._sample_var, historical_data, [])
#
#     # def generate_data(self, num_data):
#     #     data_nogradient = HistoricalData(self._dim, 0)
#     #     data_gradient = HistoricalData(self._dim, 1)
#     #     data_nogradient.append_historical_data(numpy.array([numpy.array([bound[0] for bound in self._search_domain]),
#     #                                                         numpy.array([bound[1] for bound in self._search_domain])]),
#     #                                            numpy.zeros((2,1)), numpy.ones(2) * self._sample_var)
#     #     data_gradient.append_historical_data(numpy.array([numpy.array([bound[0] for bound in self._search_domain]),
#     #                                                       numpy.array([bound[1] for bound in self._search_domain])]),
#     #                                          numpy.zeros((2,2)), numpy.ones(2) * self._sample_var)
#     #
#     #     #gp_nogradient = cppGaussianProcess(self._cov, numpy.ones(1) * self._sample_var, data_nogradient, [])
#     #     gp_gradient = cppGaussianProcess(self._cov, numpy.ones(2) * self._sample_var, data_gradient, [0])
#     #
#     #     python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])
#     #     points = python_search_domain.generate_grid_points_in_domain(num_data)
#     #
#     #     for pt in points:
#     #         val = gp_gradient.sample_point_from_gp(pt, self._sample_var)
#     #         print pt, val
#     #         data_gradient.append_sample_points([SamplePoint(pt, val, self._sample_var)])
#     #         data_nogradient.append_sample_points([SamplePoint(pt, val[0], self._sample_var)])
#     #         #gp_nogradient = cppGaussianProcess(self._cov, numpy.ones(1) * self._sample_var, data_nogradient, [])
#     #         gp_gradient = cppGaussianProcess(self._cov, numpy.ones(2) * self._sample_var, data_gradient, [0])
#     #     return data_nogradient, data_gradient
#
#     def evaluate_true(self, point, gradient=False):
#         point = numpy.array(list(point))
#         mu = self._gp.compute_mean_of_points(point)
#         grad_mu = self._gp.compute_grad_mean_of_points(point)
#         result = numpy.concatenate((mu.flatten(), [grad_mu.flatten()[0]]))
#         return result
#
#     def evaluate(self, point, gradient=False):
#         t = self.evaluate_true(point)
#         results = []
#         for r in t:
#             n = numpy.random.normal(0, numpy.sqrt(self._sample_var))
#             results += [r+n]
#         return results
