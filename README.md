# Brenier This repository implements potential flows paramterized using input convex rational quadratic (ICRQ) splines. Potential flows are a flow-based architecture for Brenier potentials that provide the Brenier map (optimal transport map) between two absolutely continuous measures.

Potential flows can be used to parameterize a $\mathbb{R}^d \to \mathbb{R}$ convex function. Two different methods for learning the Brenier optimal (an Brenier map equivalently) have been considerd:

1. Solving the Kantorovich semi-dual problem directy.
2. Solving the saddlepoint optimization problem that approximates Kantorovich semi-dual problem.

The ICRQ spline based parameterization for potential flows garners digonal transport maps. To capture the correlation structure in the data and to handle high dimensional examples, the source and target data can be encoded to a smaller dimensional latent space.

* `~/data/datasets` conatins all planar datasets considered for benchmarking.
* `~/transforms` contains all monotonic $\mathbb{R}^d \to \mathbb{R}^d$ transformations.
* `~/potential` conatains the $\mathbb{R}^d \to \mathbb{R}^d$ potential flow.
* `~/flow` contains the dual and minmax trainers.

To run an experiment using dual method for any dataset, for instance checkboard dataset, run
```
python ./flow/dual_main.py --source_dist our_checker --target_dist --our_checker 
```


To run an experiment using dual method for any dataset, for instance checkboard dataset, run
```
python ./flow/minmax_main.py --source_dist our_checker --target_dist --our_checker 
```
