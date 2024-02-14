import numpy as np

from bstrapping.synthetic_time_series.garch11 import GARCH11


class MovingAverageGARCH11:
    def __init__(self,
                 mean: float,
                 parameters: np.ndarray,
                 alpha0: float,
                 alpha: float,
                 beta: float,
                 ):
        self.mean = mean
        self.parameters = np.concatenate(([1], parameters))
        self.samples = []
        self.garch11 = GARCH11(alpha0=alpha0,
                               alpha=alpha,
                               beta=beta)

    def generate_samples(self, number_samples: int) -> np.ndarray:
        noise_terms = self.garch11.generate_samples(
            number_samples=number_samples + len(self.parameters) - 1)
        q = len(self.parameters)

        self.samples = np.array([self.mean + np.sum([self.parameters[j] * noise_terms[i - j] for j in range(q)])
                                 for i in range(q - 1, q - 1 + number_samples)])

        return self.samples

    @property
    def asymptotic_variance(self) -> float:
        # limit var(1/n^1/2*sum X_i)
        q = len(self.parameters)

        return self.garch11.variance*np.sum(
            [self.parameters[i] * self.parameters[i + np.abs(j)] for j in range(-q, q) for i in range(q - np.abs(j))])

    @property
    def variance(self) -> float:
        return self.garch11.variance*np.sum(self.parameters ** 2)
