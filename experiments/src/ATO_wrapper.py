import time
import subprocess
import random
import numpy

def convertXtoString(x):
    '''
    Matlab style string representation of x
    :param x: design point
    :return: Matlab style string representation of x
    '''
    bVector = "["
    for i in xrange(8):
        bVector += str(x[i]) + " "
    bVector += "]"
    return bVector

class Robot(object):
    def __init__(self):
        self._dim = 8
        self._search_domain = numpy.repeat([[0.0, 20.0]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """
        # Run the ATO simulator
        # b_vector is currently a string, but you can adapt it to take whatever type of array you use
        # simulation_length is a positive int that gives the length of the simulation
        # random_seed should be a random int larger zero
        # return the mean, (the variance, and the elapsed time)
        :param IS: index of information source, 1, ..., M
        :param x: 8d numpy array
        :return: the obj value at x estimated by info source IS
        """
        IS = 0

        _pathToMATLABScripts = '/fs/home/jw926/TwoStage/experiments/src/assembleToOrderExtended/'

        prg = random.Random()

        random.seed(12345)
        random_seed = prg.randint(1,100000) 			# a random int that serves as seed for matlab

        fn = -1.0
        FnVar = -1.0
        elapsed_time = 0.0

        runcmd = "b=" + convertXtoString(x) + ";length=" + str(10) + ";seed=" + str(
            random_seed)
        # print "runcmd="+runcmd

        if IS == 1:
            runcmd += ";run(\'" + _pathToMATLABScripts + "ATOHongNelson_run.m\');exit;"
        else:
            runcmd += ";run(\'" + _pathToMATLABScripts + "ATO_run.m\');exit;"

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows
            stdout = subprocess.check_output(["/usr/local/matlab/2015b/bin/matlab", "-nodisplay", "-nojvm",
                                              "-nosplash", "-nodesktop", "-r", runcmd])
            elapsed_time = time.time() - start_time

            posfn = stdout.find("fn=") + 3
            posFnVar = stdout.find("FnVar=") + 6
            if ((posfn > 2) and (posFnVar > 5)):
                posfnEnd = stdout.find("\n",posfn)
                posFnVarEnd = stdout.find("\n",posFnVar)
                fn = stdout[posfn:posfnEnd]
                FnVar = stdout[posFnVar:posFnVarEnd]
        except subprocess.CalledProcessError, e:
            elapsed_time = time.time() - start_time

        return numpy.array([-float(fn)])   # return only the mean # self._mult * float(fn)

    def evaluate(self, x):
        return self.evaluate_true(x)

class ATO(object):
    def __init__(self):
        self._dim = 8
        self._search_domain = numpy.repeat([[0.0, 20.0]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0

    def evaluate_true(self, x):
        """
        # Run the ATO simulator
        # b_vector is currently a string, but you can adapt it to take whatever type of array you use
        # simulation_length is a positive int that gives the length of the simulation
        # random_seed should be a random int larger zero
        # return the mean, (the variance, and the elapsed time)
        :param IS: index of information source, 1, ..., M
        :param x: 8d numpy array
        :return: the obj value at x estimated by info source IS
        """
        IS = 0

        _pathToMATLABScripts = '/fs/home/jw926/TwoStage/experiments/src/assembleToOrderExtended/'

        prg = random.Random()

        random.seed(12345)
        random_seed = prg.randint(1,100000) 			# a random int that serves as seed for matlab

        fn = -1.0
        FnVar = -1.0
        elapsed_time = 0.0

        runcmd = "b=" + convertXtoString(x) + ";length=" + str(10) + ";seed=" + str(
            random_seed)
        # print "runcmd="+runcmd

        if IS == 1:
            runcmd += ";run(\'" + _pathToMATLABScripts + "ATOHongNelson_run.m\');exit;"
        else:
            runcmd += ";run(\'" + _pathToMATLABScripts + "ATO_run.m\');exit;"

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows
            stdout = subprocess.check_output(["/usr/local/matlab/2015b/bin/matlab", "-nodisplay", "-nojvm",
                                              "-nosplash", "-nodesktop", "-r", runcmd])
            elapsed_time = time.time() - start_time

            posfn = stdout.find("fn=") + 3
            posFnVar = stdout.find("FnVar=") + 6
            if ((posfn > 2) and (posFnVar > 5)):
                posfnEnd = stdout.find("\n",posfn)
                posFnVarEnd = stdout.find("\n",posFnVar)
                fn = stdout[posfn:posfnEnd]
                FnVar = stdout[posFnVar:posFnVarEnd]
        except subprocess.CalledProcessError, e:
            elapsed_time = time.time() - start_time

        return numpy.array([-float(fn)])   # return only the mean # self._mult * float(fn)

    def evaluate(self, x):
        return self.evaluate_true(x)
