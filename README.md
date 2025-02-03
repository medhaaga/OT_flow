# Brenier Potential Flow

This repository implements potential flows parameterized using input convex rational quadratic (ICRQ) splines. 

## Overview
Potential flows are a flow-based architecture for estimating Brenier potential functions. The optimal transport map, aka Brenier map, between two absolutely continuous measures is obtained as the derivative of the estimated Brenier potential.
Potential flows can be used to parameterize a $\mathbb{R}^d \to \mathbb{R}$ convex function. Two different methods for learning the Brenier optimal (Brenier map equivalently) have been considered:

1. Solving the Kantorovich semi-dual problem directly.
2. Solving the saddlepoint optimization problem that approximates Kantorovich semi-dual problem.

Our ICRQ spline-based parameterization for potential flows estimates any diagonal optimal transport map by directly solving the semi-dual objective. The source and target data can be encoded to a smaller dimensional latent space to capture the correlation structure in the data and to handle high dimensional examples.

## Python Dependencies

The code mainly used PyTorch machine learning framework. The package dependencies can be found in `environment.yml`. Use the following code for `conda` environment setup.

```
conda create -n potential-flow python=3.6
conda env update -n potential-flow -f environment.yml
conda activate potential-flow
```

## Organization

* `~/data/datasets` contains all planar datasets considered for benchmarking.
* `~/transforms` contains all monotonic $\mathbb{R}^d \to \mathbb{R}^d$ transformations.
* `~/potential` conatains the $\mathbb{R}^d \to \mathbb{R}^d$ potential flow.
* `~/flow` contains the dual and minmax trainers.

To run an experiment using dual method for any dataset, for instance checkerboard dataset, run
```
python ./flow/dual_main.py --source_dist our_checker --target_dist --our_checker 
```


To run an experiment using dual method for any dataset, for instance checkboard dataset, run
```
python ./flow/minmax_main.py --source_dist our_checker --target_dist --our_checker 
```
