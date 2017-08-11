import numpy as np
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
from math_utils import * # math_utils contains all the math formulas we use

# Optimization settings
import argparse
parser = argparse.ArgumentParser(description='Find Optimal Embedding Given a Graph.')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--dim', type=int, default=2, metavar='DIM',
                    help='The embedding dimension. (default=2)')
parser.add_argument('--num_updates', type=int, default=5000, metavar='UP',
                    help='The total number of updates. (default=5000)')

def l2_loss(embedding, tree, dist_function):
    """
    Calculate the l2 loss function between space distance tensor and tree distance tensor.
    l2_loss = (space_dist - graph_dist).pow(2).sum()
    
    Inputs:
        tree         - a matrix (list or numpy array) with shape = (num_vertices, num_vertices).
                        0 on the diagonal and its a symmetric matrix.
        embedding     - The embedding tensor with shape (num_vertices, embedding_dim).
        dist_function - Distance function used for different space.
                        euclid_dist(x, y) = sqrt(x^2 +y^2)
                        hyp_dist(u, v) = arcosh(1 + 2*|u - v|^2 / ((1 - |u|^2) * (1 - |v|^2)))     
    Outputs:
        loss - FloatTensor Variable with shape (1).
    """    
    # split tensor shape = (num_vertices, dim) into num_vertices number of tensors shape = (dim).
    embedding_tuple = torch.split(embedding, 1)    
    
    # loss function is the sum of l2 norm (no sqrt) between the space distance and tree distance        
    loss = Variable(torch.FloatTensor(torch.zeros(1)))

    # calculate the distance between embedding vectors and minus the tree distance
    dist_tensor = []
    for i_idx, i in enumerate(embedding_tuple):
        for j_idx, j in enumerate(embedding_tuple):
            if i_idx <= j_idx: # when i_idx==j_idx (dist=0) as it will lead to NaN loss in backprop
                continue
            dist_tensor.append((dist_function(i,j) - tree[i_idx][j_idx]).pow(2))

    # stack the list of calculated distance
    dist_tensor = torch.stack(dist_tensor)

    # loss = L2 loss between space distance tensor and tree distance tensor
    loss = dist_tensor.sum()
    
    return loss

def findOptimalEmbedding(tree, embedding_dim=2, lr=1e-3, num_updates=5000):
    """
    Given a tree (or even graph) with its distance defined in a matrix, 
    find the optimal embedding in Euclidean space and hyperbolic space.
    
    Inputs:
        graph - a matrix (list or numpy array) with shape = (num_vertices, num_vertices).
                0 on the diagonal and its a symmetric matrix.
    Outputs:
        saves euclidean embedding - shape = (num_vertices, dim). Default dimension = 2 for visualization.
        saves hyperbolic embedding - shape = (num_vertices, dim). Default dimension = 2 for visualization.
        
    TODO:
        -Sometimes it get stuck in local minima, need to implement momentum.
        -Unit tests for hyp_dist and inverse_metric_tensor.
    """
    num_vertices = len(tree)    
    
    # initialize euclidean embedding tensor ~ uniform distribution in range [0, 1)
    euclid_embedding = Variable(torch.rand(num_vertices, embedding_dim).type(torch.FloatTensor), requires_grad=True)
    
    # initialize euclidean embedding tensor ~ uniform distribution in range [0, 0.1)
    hyp_embedding = Variable(torch.div(torch.rand(num_vertices, embedding_dim), 10).type(torch.FloatTensor), requires_grad=True)
    
    print('Finding Optimal Embedding with dim = %i, lr = %f, total number of updates = %i' %(embedding_dim, lr, num_updates))
    for t in range(num_updates):
        
        # l2_loss function is the sum of l2 norm (no sqrt) between the space distance and tree distance        
        euclid_loss = l2_loss(euclid_embedding, tree, euclid_dist)
        hyp_loss = l2_loss(hyp_embedding, tree, hyp_dist)
        
        # print out loss in console
        sys.stdout.write('\r' + ('%i: euclid loss = %f, hyperbolic loss = %f' % (t, euclid_loss.data[0],  hyp_loss.data[0])))
        sys.stdout.flush() 
        
        # using autograd, get gradients for embedding tensors
        euclid_loss.backward()
        hyp_loss.backward()
        
        # Update weights using gradient descent
        euclid_embedding.data -= lr * euclid_embedding.grad.data
        hyp_embedding.data -= lr *inverse_metric_tensor(hyp_embedding)*hyp_embedding.grad.data

        # Manually zero the gradients after updating weights
        euclid_embedding.grad.data.zero_()
        hyp_embedding.grad.data.zero_()        
        
    print('\n finished optimization!')
    np.save('euclid_embedding.npy', euclid_embedding.data.numpy())
    np.save('hyp_embedding.npy', hyp_embedding.data.numpy())
    print('Saved Euclidean embedding to euclidean_embedding.npy and hyperbolic embedding to hyp_embedding.npy !')

if __name__=="__main__":
    
    args = parser.parse_args()    
                           
    #    weighted tree adjacency matrix
    #    A  B  C  D  E  F  G  H  I  J  K
    A = [0, 1, 2, 2, 1, 2, 2, 3, 3, 1, 2]
    B = [1, 0, 1, 1, 2, 3, 3, 4, 4, 2, 1]
    C = [2, 1, 0, 2, 3, 4, 4, 5, 5, 3, 2]
    D = [2, 1, 2, 0, 3, 4, 4, 5, 5, 3, 2]
    E = [1, 2, 3, 3, 0, 1, 1, 2, 2, 2, 3]
    F = [2, 3, 4, 4, 1, 0, 2, 3, 3, 3, 4]
    G = [2, 3, 4, 4, 1, 2, 0, 1, 1, 3, 4]
    H = [3, 4, 5, 5, 2, 3, 1, 0, 2, 4, 5]
    I = [3, 4, 5, 5, 2, 3, 1, 2, 0, 4, 5]
    J = [1, 2, 3, 3, 2, 3, 3, 4, 4, 0, 3]
    K = [2, 1, 2, 2, 3, 4, 4, 5, 5, 3, 0]
    tree = [A, B, C, D, E, F, G, H, I, J, K] 
    
    findOptimalEmbedding(tree, args.dim, args.lr, args.num_updates)
                           
                           
                           