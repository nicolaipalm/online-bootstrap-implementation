import numpy as np
from arch.univariate import GARCH as model


class GARCH:
    def __init__(self,
                 alpha0: float,
                 alpha: np.ndarray,
                 beta: np.ndarray,
                 ):
        if sum(alpha) + sum(beta) >= 1:
            raise ValueError('The sum of alpha and beta must be strictly smaller than 1.')
        self._variance = alpha0 / (1 - np.sum(alpha) - np.sum(beta))
        self.samples = []
        self.alpha0 = alpha0
        self.alpha = alpha
        self.beta = beta

    def generate_samples(self,
                         number_samples: int,) -> np.ndarray:
        # GARCH is a special case of ARMA, i.e. strictly stationary and stongly mixing
        # Source: GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY - Tim BOLLERSLEV
        # epsilon_1 = N(0,sigma^2)
        # E(epsilon_t) = mean, var(epsilon_t) = a_0(1-sum_{i=1}^q a_i- sum_{i=1}^p b_i)^{-1} = sigma^2
        # =here= parameter/(1-(q+p)*parameter) i.e. parameter = sigma^2/(1+(q+p)*sigma^2)
        # epsilon_t|psi ~ N(mean,h_t)
        # h_t = a_0 + sum_{i=1}^q a_i epsilon_{t-i}^2 + sum_{i=1}^p b_i * h_{t-1}
        #     =here= parameter(1+sum_{i=1}^q epsilon_{t-i}^2 + sum_{i=1}^p h_{t-1})

        q = len(self.alpha)
        p = len(self.beta)
        garch = model(p=p, q=q, )

        parameters = np.concatenate((np.array([self.alpha0]), self.alpha, self.beta))  # parameters = [alpha0, alpha, beta]
        print(parameters)
        self.samples = garch.simulate(parameters=parameters,
                                      nobs=number_samples,
                                      rng=lambda x: np.random.normal(loc=0, scale=1, size=x))[0] + self.mean
        return self.samples

    @property
    def asymptotic_variance(self) -> float:
        return self._variance

    @property
    def variance(self) -> float:
        return self._variance
