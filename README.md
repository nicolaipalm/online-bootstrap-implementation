# An online bootstrap for time series - Implementation
This module contains the implementation of the AR online bootstrap proposed in [this paper](https://arxiv.org/abs/2310.19683) and its benchmark
against the standard iid bootstrap and the (offline) MA bootstrap for time series.
The latter may be found in the benchmarking submodule.

## Installation

You can clone this repository and install its requirements by running the following commands:

```shell
git clone git@github.com:nicolaipalm/online-bootstrap-implementation.git
cd online-bootstrap-implementation
pip install -r requirements.txt
```

## A short summary of bootstrapping

Bootstrapping methods are re-sampling schemes with (asymptotic) theoretical guarantees.
Given some (random) samples according to some distribution, bootstrapping allows you to
generate (computationally cheap) synthetic new samples.
Those samples behave similar in the following sense:
The distribution of the *arithmetic mean* of the synthetic samples is (asymptotically)
the distribution of the *arithmetic mean* of the real samples.
Accordingly, we can use the synthetic samples in order to approximate the distribution of the
arithmetic mean of the real samples.



