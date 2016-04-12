import numpy as np


def quasiper_kernel(alpha, dx):
    """
    The quasiperiodic kernel function. The difference matrix
    can be an arbitrarily shaped numpy array so make sure that you
    use functions like ``numpy.exp`` for exponentiation.
    
    :param alpha: ``(4,)`` The parameter vector ``(amplitude, decay time,
    period, structure param)``.
    
    :param dx: ``numpy.array`` The difference matrix. This can be
        a numpy array with arbitrary shape.
    
    :returns K: The kernel matrix (should be the same shape as the
        input ``dx``). 
    
    """
    return alpha[0]**2 * np.exp(-0.5 * dx**2 / alpha[1]**2 - 2.0 *
                                np.sin((np.pi * dx / alpha[2]))**2 /
                                alpha[3]**2)
