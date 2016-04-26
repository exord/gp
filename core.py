import numpy as np
from scipy.linalg import cho_factor, cho_solve

import kernels


def sample_gp(x, gp=None, alpha=np.empty(0), kerneltype='se', size=1):
    """
    :param array x: input array where GP is to be sampled.
    :param gp: a `~gp.core.GaussianProcess` class instance. If None,
    the instance will be constructed using hyperparameter vector alpha and
    kerneltype.
    :param array-like alpha: List of hyperparameters to build kernel instance.
    :param string kerneltype: The kernel type. Options are:
        'se' for a squared exponential kernel.
        'qper' for a quasiperiodic kernel.
    :param int size: size of the sample.
    """
    # If a Kernel instance is passed.
    if isinstance(gp, GaussianProcess):
        gp.set_test_input(x)
        return gp.sample(size)
    # Otherwise, construct instance on the fly.
    elif kerneltype == 'se':
        return kernels.SquaredExponentialKernel(alpha).sample(x, size)
    elif kerneltype == 'ge':
        return kernels.GeneralisedExponentialKernel(alpha).sample(x, size)
    elif kerneltype == 'qper':
        return kernels.QuasiPeriodicKernel(alpha).sample(x, size)
    else:
        raise NameError('Kerneltype not recognised.')


class GaussianProcess(object):
    """
    A Class implementing Gaussian processes.
    """

    def __init__(self, kernel, xinput, data=None):
        """
        :param kernel: an instance of the :class:`~gp.kernels.Kernel`
        :param np.array xinput: "test" input coordinates.
        :param np.array data: a `(N x 2)` or `(N x 3)` array of N data inputs:
         (data coordiante, data value, data error (optional)).
        """
        # Initialise input attributes (for PEP-8 compliance).
        self._input = None
        self._data = None
        self.covariance = None
        self.covariance_data = None
        self.covariance_test_data = None

        # Set kernel
        self.kernel = kernel

        # Set the input test coordinates
        self.x = xinput

        # Set data (if given).
        self.data = data

    @property
    def x(self):
        """
        The GP test input coordinate vector.
        """
        return self._input

    @x.setter
    def x(self, inputarray):
        self._input = inputarray

        # Compute the diffrence matrix and covariance
        dx = self._input[:, None] - self._input[None, :]
        self.covariance = self.kernel.covariance(dx)

        if self.data is not None:
            cov_star_data, cov_data = self.computecovariances(self._data)
            self.covariance_test_data = cov_star_data

    @x.deleter
    def x(self):
        self._input = None
        self.covariance = None

    def get_test_input(self):
        return self.x

    def set_test_input(self, inputarray):
        self.x = inputarray

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dataarray):
        if dataarray is not None:
            self._data = dataarray
            cov_star_data, cov_data = self.computecovariances(self._data)
            self.covariance_test_data = cov_star_data
            self.covariance_data = cov_data

    @data.deleter
    def data(self):
        self.data = None
        self.covariance_data = None
        self.covariance_test_data = None

    def erasedata(self):
        del self.data

    def computecovariances(self, data):
        """
        Compute the covariances between the data inputs (data) and the test
        inputs (star).

        :param np.array data: a 2-D array with dimensions (2, n) or (3, n).
        :returns: covariances matrices
        """
        xdata = data[0]
        dx_star_data = self.x[:, None] - xdata[None, :]
        dx_data = xdata[:, None] - xdata[None, :]
        return self.kernel.covariance(dx_star_data), self.kernel.covariance(
            dx_data)

    def sample(self, size=1):
        return np.random.multivariate_normal(np.zeros_like(self.x),
                                             self.covariance, size)

    def prediction(self, data=None):

        if data is None and self.data is None:
            raise TypeError('Data array cannot be None, unless you want your'
                            'predictions to look like your prior. In that'
                            'case, better use the `sample` method.')

        elif data is not None:

            if self.data is not None:
                print('Data given. Overriden previous data.')
            self.data = data

            # Compute covariance matrices
            cov_test_data, cov_data = self.computecovariances(self.data)
            self.covariance_test_data = cov_test_data
            self.covariance_data = cov_data

        # If errors are provided for data, add them to the covariance diagonal
        if self.data.shape[0] > 2:
            dataerror = np.diag(self.data[2] ** 2)
        else:
            dataerror = np.diag(np.zeros_like(self.data[0]))

        # Use Cholesky decomposition on covariance of data inputs.
        factor, flag = cho_factor(self.covariance_data + dataerror)

        # Compute posterior mean (eq. 2.23 Rasmussen)
        a = cho_solve((factor, flag), self.data[1])
        self.predmean = np.dot(self.covariance_test_data, a)

        # Compute posterior covariance (eq. 2.24 Rasmussen)
        alpha = cho_solve((factor, flag), self.covariance_test_data.T)
        beta = np.dot(self.covariance_test_data, alpha)
        self.predcov = self.covariance - beta


__author__ = 'Rodrigo F. Diaz'
