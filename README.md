# Brenier Potential Flow

This repository implements potential flows parameterized using input convex rational quadratic (ICRQ) splines. Potential flows are a flow-based architecture for estimating Brenier potential functions. The optimal transport map, aka Brenier map, between two absolutely continuous measures is obtained as the derivative of the estimated Brenier potential.

Potential flows can be used to parameterize a $\mathbb{R}^d \to \mathbb{R}$ convex function. Two different methods for learning the Brenier optimal (Brenier map equivalently) have been considered:

1. Solving the Kantorovich semi-dual problem directly.
2. Solving the saddlepoint optimization problem that approximates Kantorovich semi-dual problem.

The ICRQ spline-based parameterization for potential flows garners diagonal transport maps. The source and target data can be encoded to a smaller dimensional latent space to capture the correlation structure in the data and handle high dimensional examples.

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
