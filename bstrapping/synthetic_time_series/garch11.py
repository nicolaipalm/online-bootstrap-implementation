import numpy as np


class GARCH11:
    def __init__(self,
                 alpha0: float,
                 alpha: float,
                 beta: float,
                 mean: float = 0,
                 ):
        if alpha + beta >= 1:
            raise ValueError('The sum of alpha and beta must be strictly smaller than 1.')
        self._variance = alpha0 / (1 - alpha - beta)
        self.samples = []
        self.alpha0 = alpha0
        self.alpha = alpha
        self.beta = beta
        self.h = None

    def generate_samples(self,
                         number_samples: int, ) -> np.ndarray:
        # GARCH is a special case of ARMA, i.e. strictly stationary and strongly mixing
        # epsilon_t|psi ~ N(mean,h_t)
        # h_t = a_0 + sum_{i=1}^q a_i epsilon_{t-i}^2 + sum_{i=1}^p b_i * h_{t-1}
        # Source: GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY - Tim BOLLERSLEV
        # GARCH model is uncorrelated and strictly stationary whenever alpha_0/(1-sum(alpha)-sum(beta))<1 with
        # E(epsilon_t) = mean
        # var(epsilon_t) = a_0(1-sum_{i=1}^q a_i- sum_{i=1}^p b_i)^{-1} = sigma^2

        # initialize zero array for samples
        n1 = 1000  # drop the first n1 observations
        n2 = number_samples + n1

        # a GARCH model is a priori not well-defined since it is obtained by recursion with unspecified initial values
        # however, the first observations don't matter since the GARCH model stabilizes
        # we choose the first observation as standard normal (i.e. h_1=1) and drop the first n1 observations afterward
        mu = np.random.normal(0, 1, n2)  # mu iid normal(0,1) distributed
        self.samples = np.zeros(n2)  # = (epsilon_t)
        self.h = np.zeros(n2)

        self.h[0] = 1
        self.samples[0] = np.sqrt(self.h[0]) * mu[0]

        # h_t = alpha_0+alpha_1*epsilon_{t-1}^2+beta_1*h_{t-1}
        # epsilon_t = h_t^1/2*mu_t
        for i in range(1, n2):
            self.h[i] = self.alpha0 + self.alpha * self.samples[i - 1] ** 2 + self.beta * self.h[i - 1]
            self.samples[i] = np.sqrt(self.h[i]) * mu[i]

        self.samples = self.samples[n1:]  # drop the first n1 observations

        return self.samples

    @property
    def asymptotic_variance(self) -> float:
        return self._variance

    @property
    def variance(self) -> float:
        return self._variance
