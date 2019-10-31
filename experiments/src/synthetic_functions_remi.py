import numpy
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import math

class Branin(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[0, 15], [-5, 15]])
        self._num_init_pts = 40
        self._sample_var = 0.0
        self._min_value = 0.397887
        self._observations = []
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
        self._search_domain = numpy.array([[-3., 3.], [-2., 2.]])
        self._num_init_pts = 40
        self._sample_var = 0.0
        self._min_value = -1.0316
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
            :param x[2]: 2-dim numpy array
        """
        x2 = math.pow(x[0],2)
        x4 = math.pow(x[0],4)
        y2 = math.pow(x[1],2)

        return numpy.array([(4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x[0]*x[1] + (-4.0 + 4.0 * y2) * y2])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Goldstein(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[-2., 2.], [-2., 2.]])
        self._num_init_pts = 40
        self._sample_var = 0.0
        self._min_value = 3.
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
            :param x[2]: 2-dim numpy array
        """
        term1 = 1. + pow((x[0]+x[1]+1.), 2)*(19-14*x[0]+3*pow(x[0], 2)-14*x[1]+6*x[0]*x[1]+3*pow(x[1], 2))
        term2 = 30. + pow((2*x[0]-3*x[1]), 2)*(18.-32.*x[0]+12*pow(x[0], 2)+48*x[1]-36.*x[0]*x[1]+27*pow(x[1], 2))

        return numpy.array([term1*term2])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Griewank(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[-6., 6.], [-6., 6.]])
        self._num_init_pts = 40
        self._sample_var = 0.0
        self._min_value = 0.
        self._observations = []
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
            :param x[2]: 2-dim numpy array
        """
        x *= 100.

        return numpy.array([(pow(x[0], 2) + pow(x[1], 2))/4000. - (numpy.cos(x[0]) * numpy.cos(x[1]/sqrt(2))) + 1.])

    def evaluate(self, x):
        return self.evaluate_true(x)