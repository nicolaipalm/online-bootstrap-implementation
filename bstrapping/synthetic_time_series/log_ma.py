import numpy as np


class LogMovingAverage:
    def __init__(self, parameters: np.ndarray, mu: float = 0):
        self.mu = mu
        self.parameters = np.concatenate(([1], parameters))
        self._variance_ma = np.sum(self.parameters ** 2)
        self.samples = []
        self.mean = np.exp(self.mu + 1 / 2 * self._variance_ma)  # log normal exp(mu + 1/2*sigma^2)

    def generate_samples(self, number_samples: int) -> np.ndarray:
        noise_terms = np.random.normal(loc=0, scale=1, size=number_samples + len(self.parameters))
        q = len(self.parameters)

        self.samples = np.array([np.exp(self.mu + np.sum([self.parameters[j] * noise_terms[i - j] for j in range(q)]))
                                 for i in range(number_samples)])

        return self.samples

    def _covariance_ma(self, h):
        q = len(self.parameters)
        if h > len(self.parameters):
            return 0
        else:
            return np.sum(
                [self.parameters[i] * self.parameters[i + np.abs(h)] for i in range(q - np.abs(h))])

    def _covariance_log_ma(self, h):
        # exp(mu_i + mu_j + 1/2*(sigma_i^2 + sigma_j^2))*(exp(sigma_ij)-1)
        # here: exp(2*mu +sigma^2)*exp(sigma_ij-1)
        h = np.abs(h)
        return np.exp(2 * self.mu + self._variance_ma) * (np.exp(self._covariance_ma(h)) - 1)

    @property
    def asymptotic_variance(self) -> float:
        # limit var(1/n^1/2*sum X_i)
        q = len(self.parameters)

        return np.sum(
            [self._covariance_log_ma(h) for h in range(-q, q)])

    @property
    def variance(self) -> float:
        return np.exp(self._variance_ma - 1) * np.exp(
            2 * self.mu + self._variance_ma)  # exp(sigma^2-1) * exp(2*mu + sigma^2)
