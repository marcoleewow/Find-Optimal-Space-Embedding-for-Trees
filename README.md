# Find-Optimal-Space-Embedding-for-Trees

Using pyTorch 0.1.12 and Python 2.7.

Given a weighted adjacency matrix of a connected finite tree in Euclidean space and Hyperbolic space, calculate the optimal embedding using Stochastic Gradient Descent(SGD).

NOTE:
Since the loss function w.r.t. embedding vectors is not convex, SGD does not guarantee convergence to global minima! 
Run a couple of times until you have obtain the lowest loss score.

TODO:
- [ ] add images
- [ ] find new method to guarantee convergence to global minima
