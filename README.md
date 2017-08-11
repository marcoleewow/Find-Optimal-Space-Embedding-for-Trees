# Find-Optimal-Space-Embedding-for-Trees

Using pyTorch 0.1.12 and Python 2.7.

Given a weighted adjacency matrix of a connected finite tree in Euclidean space and Hyperbolic space, calculate the optimal embedding using Stochastic Gradient Descent(SGD).

NOTE:
Since the loss function w.r.t. embedding vectors is not convex, SGD does not guarantee convergence to global minima! 
Run a couple of times until you have obtain the lowest loss score.

TODO:
- [x] add images

- [ ] find new method to guarantee convergence to global minima

## Example

Given a connected finite tree that looks like this:

![alt text][tree_1]

We can have a lot of different visualizations for this tree, since we dont care about the angles and real length of edges. 
For example:

![alt text][tree_2]

The two trees are equivalent! So a better representation of this tree is to use adjacency matrix. Let each edge to have a tree length 1, then the weighted tree adjacency matrix is:

![alt text][tree_dist_matrix]

To find an optimal embedding of this tree in Euclidean space, we transform the problem into an optimization problem so we can solve (approximate might be a better word) with SGD.

loss = L2_loss(space_distance_matrix, tree_distance_matrix)

And so now we minimise loss w.r.t. embedding vectors of vertices using SGD. 
(note: space_distance depends on embedding vectors, tree_distance stays constant)

## Euclidean Embedding

We then have:

![alt text][euclid_tree]

with loss = 7.166, and the space distance matrix look like:

![alt text][euclid_dist_matrix]

## Hyperbolic Embedding

Using Riemannian SGD, we can also find the optimal embedding in Hyperbolic space. We then have:

![alt text][hyp_tree] 

(plotted using GeoGebra [worksheet](https://www.geogebra.org/m/R5e9AggU))

with loss = 1.928, and the space distance matrix look like:

![alt text][hyp_dist_matrix]

This shows that embedding trees in Euclidean space is not as efficient as in Hyperbolic space, or in other words its easier to approximate trees in Hyperbolic space.

[tree_1]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/tree_1.png "Tree Example"
[tree_2]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/tree_2.png "Tree Example 2"
[tree_dist_matrix]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/tree_dist_matrix.png "tree_dist_matrix"
[euclid_tree]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/euclid_tree.png "euclid_tree"
[euclid_dist_matrix]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/euclid_dist_matrix.png "euclid_dist_matrix"
[hyp_tree]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/hyp_tree.png "hyp_tree"
[hyp_dist_matrix]: https://github.com/marcoleewow/Find-Optimal-Space-Embedding-for-Trees/blob/master/images/euclid_dist_matrix.png "hyp_dist_matrix"
