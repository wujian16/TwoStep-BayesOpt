import numpy
from numpy import genfromtxt

class hyper_tuning_grid(object):
    def __init__(self, dataname):
        self.dataname = dataname
        #self.datapath = '/Users/jianwu/Google Drive/TwoStepBO/experiments/Raw.Results/%s.csv' % dataname
        #self.datapath = 'data/%s.csv' % dataname
        
        if dataname == 'svm_on_grid':
            self._dim = 3
            self._min_value = 0.2411 
        elif dataname == 'lda_on_grid':
            self._dim = 3
            self._min_value = 0.0
        elif dataname == 'logreg_on_grid':
            self._dim = 4
            self._min_value = 0.0685
        
        self._search_domain = numpy.tile([0,1], (self._dim, 1))
        self._num_init_pts = 1
        self._sample_var = 0.0
        self._num_fidelity = 0
        self._observations = []
        #self.data = genfromtxt(self.datapath, delimiter=',')
        self.index = 0
        self.chosen_idx = None
        
##    def find_closest(self, x, chosen_idx):
##        # find the point in the grid closest to x in Euclidean norm
##        n = self.data.shape[0]
##        unchosen_idx = numpy.ones(n, dtype=bool)
##        if chosen_idx != None:
##            unchosen_idx[chosen_idx] = 0
##        unchosen_idx = numpy.nonzero(unchosen_idx)[0]
##        num_rem = len(unchosen_idx)
##        xx = numpy.tile(x, (num_rem, 1))
##        distance = numpy.linalg.norm(xx - self.data[unchosen_idx,:-1], axis=1)
##        #print distance
##        index = numpy.argmin(distance)
##        index = unchosen_idx[index]  # np.nonzero returns a tuple
##        # return the index of the point closest to x among the unchosen ones
##        self.index = index
##        self.chosen_idx = chosen_idx
##        return index
##        
##    def evaluate_true(self, x):
##        # if self.index:
##        #     index = self.index
##        # else:
##        index = self.find_closest(x, self.chosen_idx)
##        return self.data[index, ::-1]
##
##    def evaluate(self, x):
##        return self.evaluate_true(x)
