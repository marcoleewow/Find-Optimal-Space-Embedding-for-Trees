import numpy as np
import torch

# utils function in numpy

def np_euclid_dist(x, y):
    """
    Euclidean distance between two tensors (numpy only).
    euclid_dist(x) = sqrt(x^2 + y^2)
    """
    return np.linalg.norm(np.asarray(x)-np.asarray(y))

def np_hyp_dist(u, v):
    """
    Hyperbolic distance between two vectors (numpy only).
    hyp_dist(x) = arcosh(1 + 2*|u - v|^2 / ((1 - |u|^2) * (1 - |v|^2)))    
    """
    # make sure its in np.array type.
    u = np.asarray(u)
    v = np.asarray(v)
    
    # get norm in the denominator.
    u_norm = 1. - np.linalg.norm(u)**2
    v_norm = 1. - np.linalg.norm(v)**2 
    
    # get the delta
    delta = 2. * (np.linalg.norm(u-v)**2) / (u_norm * v_norm)
    
    # calculate hyp dist
    return np.arccosh(1. + delta)


# ----------------------------------------------------------------------------------------------------
# utils function in pyTorch

def get_distance_adj_matrix(euclid_embedding, dist_func):
    embed_list = euclid_embedding.tolist()
    num_vertices = len(euclid_embedding)
    dist = np.zeros([num_vertices, num_vertices])

    # get the distance matrix 
    for i_idx, i in enumerate(embed_list):
        for j_idx, j in enumerate(embed_list):
            if i_idx <= j_idx:
                continue
            dist[i_idx][j_idx] = dist_func(i, j)
            dist[j_idx][i_idx] = dist[i_idx][j_idx]
    
    return dist

# distance functions
def euclid_dist(x, y):
    """
    Euclidean distance between two tensors.
    euclid_dist(x) = sqrt(x^2 + y^2)
    """
    return torch.dist(x, y, p=2)

def arccosh(x):
    """
    arcosh(x) = ln(x + sqrt(x^2 - 1))
    elementwise arcosh operation.
    """
    return torch.log(x + torch.sqrt(torch.add(torch.pow(x, 2), -1.)))

def hyp_dist(u, v):
    """
    Hyperbolic distance between two tensors.
    hyp_dist(x) = arcosh(1 + 2*|u - v|^2 / ((1 - |u|^2) * (1 - |v|^2)))    
    """
    # u_norm = 1 - |u|^2
    u_norm = torch.add((torch.neg(torch.pow(torch.norm(u, 2, 1, keepdim=True), 2))), 1.)
    # v_norm = 1 - |v|^2
    v_norm = torch.add((torch.neg(torch.pow(torch.norm(v, 2, 1, keepdim=True), 2))), 1.)
    # delta is the isometric invariants, del = 2*|u - v|^2 / (u_norm * v_norm)                            
    delta = torch.mul(torch.pow(torch.dist(u, v), 2), 2.) / torch.mul(u_norm, v_norm)

    hyper_dist = arccosh(1 + delta)
    
    assert hyper_dist.size() == (1, 1)
    return hyper_dist

def inverse_metric_tensor(theta):
    """
    gives the inverse metric tensor at theta. 
    Use this to multiply euclidean gradients and convert to riemannian gradients.
    Input:
        theta - the embedding vectors, shape = (batch_size, ndim).
    Return:
        the scalar = (1 - ||theta||^2)^2 / 4, shape = (batch_size, 1).
    """
    # || theta ||
    norm = torch.norm(theta, 2, 1, keepdim=True)
    one = torch.ones(norm.size())
    norm = torch.pow(norm, 2)

    # 1 - ||theta||^2
    ret = torch.ones(norm.size()) - torch.pow(norm.data, 2) # note the data here, use tensor instead of variable

    # (1 - ||theta||^2)^2 / 4
    return torch.div(torch.pow(ret, 2), 4.)