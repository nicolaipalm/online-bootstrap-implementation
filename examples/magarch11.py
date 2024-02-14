import numpy as np
from matplotlib import pyplot as plt

from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap
from bstrapping.bootstrap_procedures.online_ar_bootstrap import OnlineARBootstrap
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
from bstrapping.synthetic_time_series.magarch11 import MovingAverageGARCH11

from bstrapping.weights.moving_average import MovingAverageWeights

sample_size = 10000

mean = 4
alpha0 = 1
alpha = 0.25
beta = 0.25
parameters = np.array([0.5 ** i for i in range(1, 3)])

magarch = MovingAverageGARCH11(mean=mean, parameters=parameters, alpha0=alpha0, alpha=alpha, beta=beta)
samples = magarch.generate_samples(number_samples=sample_size)

plt.title(f'GARCH ({1, 1}) process')
x = range(sample_size)
plt.plot(x, samples)
plt.show()

print(magarch.asymptotic_variance)

bootstrap = OnlineARBootstrap()
bootstrap(new_samples=samples, number_bootstrap_samples=250)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance online: \n {bootstrap.estimated_asymptotic_variance}')
print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')

# Perform the discrete bootstrap
bootstrap = DiscreteBootstrap(samples=samples, number_bootstrap_samples=250)

# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance iid: \n {bootstrap.estimated_asymptotic_variance}')

print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')

weights = MovingAverageWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights, number_bootstrap_samples=250)
print(f'Bootstrapped variance block-bootstrap: \n {bootstrap.estimated_asymptotic_variance}')
print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')
