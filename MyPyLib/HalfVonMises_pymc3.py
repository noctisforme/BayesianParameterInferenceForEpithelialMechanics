# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 00:08:37 2022

@author: Goshi
"""

import theano.tensor as tt
import scipy.stats as stats
import numpy as np

from pymc3.distributions import transforms
from pymc3.distributions.continuous import assert_negative_support
from pymc3.distributions.distribution import Continuous, draw_values, generate_samples
from pymc3.theanof import floatX
from pymc3.distributions.special import log_i0
from pymc3.distributions.dist_math import bound



class Polar(transforms.ElemwiseTransform):
    name = "polar"

    def backward(self, y):
        z = tt.arctan2(tt.sin(2*y), tt.cos(2*y))/2
        return tt.switch(z < 0, z + np.pi, z)
    
    def forward(self, x):
        return tt.as_tensor_variable(x)

    def forward_val(self, x, point=None):
        return x

    def jacobian_det(self, x):
        return tt.zeros(x.shape)

class HalfVonMises(Continuous):
    r"""
    Univariate VonMises log-likelihood.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-np.pi, np.pi, 200)
        mus = [0., 0., 0.,  -2.5]
        kappas = [.01, 0.5,  4., 2.]
        for mu, kappa in zip(mus, kappas):
            pdf = st.vonmises.pdf(x, kappa, loc=mu)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\kappa$ = {}'.format(mu, kappa))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    Variance  :math:`1-\frac{I_1(\kappa)}{I_0(\kappa)}`
    ========  ==========================================

    Parameters
    ----------
    mu: float
        Mean.
    kappa: float
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).
    """

    def __init__(self, mu=0.0, kappa=None, transform="polar", *args, **kwargs):
        if transform == "polar":
            transform = Polar()
        super().__init__(transform=transform, *args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.kappa = kappa = tt.as_tensor_variable(floatX(kappa))

        assert_negative_support(kappa, "kappa", "HalfVonMises")

    def random(self, point=None, size=None):
        """
        Draw random values from VonMises distribution.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, kappa = draw_values([self.mu, self.kappa], point=point, size=size)
        return generate_samples(
            0.5 * stats.vonmises.rvs, loc=mu, kappa=kappa, dist_shape=self.shape, size=size
        )

    def logp(self, value):
        """
        Calculate log-probability of VonMises distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        kappa = self.kappa
        return bound(
            kappa * tt.cos(2*mu - 2*value) - (tt.log(2 * np.pi) + log_i0(kappa)),
            kappa > 0,
            value >= 0,
            value <= np.pi,
        )

    def _distr_parameters_for_repr(self):
        return ["mu", "kappa"]

class PolarUniform(Continuous):
    
    def __init__(self, transform="polar", *args, **kwargs):
        if transform == "polar":
            transform = Polar()
        super().__init__(transform=transform, *args, **kwargs)
        self.mean = np.pi/2
        self.median = self.mean

        assert_negative_support("PolarUniform","","PolarUniform")

    def random(self, point=None, size=None):
        
        return generate_samples(
            stats.uniform.rvs, loc=0, scale=np.pi, dist_shape=self.shape, size=size
        )

    def logp(self, value):
        """
        Calculate log-probability of VonMises distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        return bound(
           -tt.log(np.pi),
           value >= 0,
           value <= np.pi,
        )
