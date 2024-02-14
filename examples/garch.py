from matplotlib import pyplot as plt

from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap
from bstrapping.bootstrap_procedures.online_ar_bootstrap import OnlineARBootstrap
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
from bstrapping.synthetic_time_series.garch11 import GARCH11

from bstrapping.weights.moving_average import MovingAverageWeights

sample_size = 3000

alpha0 = 2
alpha = 0.4
beta = 0.4

# garch = GARCH(mean=4, alpha0=2, alpha=alpha, beta=beta)
garch = GARCH11(alpha0=alpha0, alpha=alpha, beta=beta)
samples = garch.generate_samples(number_samples=sample_size)

plt.title(f'GARCH ({1,1}) process')
x = range(sample_size)
plt.plot(x, samples)
plt.show()

print(garch.variance)

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
