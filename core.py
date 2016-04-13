import numpy as np
import kernels


def sample_gp(x, gp=None, alpha=np.empty(0), kerneltype='se', size=1):
    """
    :param array x: input array where GP is to be sampled.
    :param Kernel gp: a Kernel class instance. If None, the Kernel will be
    constructed using parameter alpha and kerneltype.
    :param array-like alpha: List of hyperparameters to build kernel instance.
    :param string kerneltype: The kernel type. Options are:
        'se' for a squared exponential kernel.
        'qper' for a quasiperiodic kernel.
    :param int size: size of the sample.
    """
    # If a Kernel instance is passed.
    if isinstance(gp, kernels.Kernel):
        return gp.sample(x, size)
    # Otherwise, construct instance on the fly.
    elif kerneltype == 'se':
        return kernels.SquaredExponentialKernel(alpha).sample(x, size)
    elif kerneltype == 'qper':
        return kernels.QuasiPeriodicKernel(alpha).sample(x, size)
    else:
        raise NameError('Kerneltype not recognised.')


__author__ = 'Rodrigo F. Diaz'
