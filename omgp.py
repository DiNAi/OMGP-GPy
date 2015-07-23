import GPy
import numpy as np

class OMGP(GPy.core.GP):
	""" OMGP
	"""
    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None):

        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1])

        likelihood = GPy.likelihoods.Gaussian()

        super(GPRegression, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer)

        self.num_models = 2

        # Initiate responsibilities
        self.qZ = np.random.rand(self.num_data, self.num_models)
		self.qZ = (self.qZ.T / self.qZ.sum(1)).T

		self.logqZ = np.log(self.qZ)
		self.logqZ = self.logqZ - np.outer(self.logqZ[:, 0], np.ones((1, 2)))
		self.logqZ = self.logqZ[:, 1:]

	def E_inc(self):
		""" Incremental E-step in the EM algorithm.
		"""

		maxit = self.num_data + 100

		for i in range(maxit):
			sqB 