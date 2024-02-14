import numpy as np

from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap
from bstrapping.bootstrap_procedures.weighted_bootstrap import WeightedBootstrap
from bstrapping.synthetic_time_series.log_ma import LogMovingAverage
from bstrapping.weights.auto_regressive_weights import AutoRegressiveWeights
from bstrapping.weights.moving_average import MovingAverageWeights

# specify variance, mean and number of the samples

number_sample_points = 5000

# sample
parameters = np.array([0.5 ** i for i in range(1, 3)])
process = LogMovingAverage(mu=0, parameters=parameters)

samples = process.generate_samples(number_samples=number_sample_points)
print(f'Empirical mean: {np.mean(samples)}, real mean: {process.mean}')

# calculated variance
true_variance = process.asymptotic_variance
true_mean = process.mean
print(f'True variance of empirical mean ({true_mean}): {true_variance}')

# Perform the discrete bootstrap
bootstrap = DiscreteBootstrap(samples=samples, number_bootstrap_samples=250)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.estimated_asymptotic_variance}')
print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')

# Perform the weighted bootstrap
weights = AutoRegressiveWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights, number_bootstrap_samples=250)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.estimated_asymptotic_variance}')
print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')

weights = MovingAverageWeights(samples=samples)
bootstrap = WeightedBootstrap(samples=samples, weights=weights)
# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.estimated_asymptotic_variance}')
print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')
