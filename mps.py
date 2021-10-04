import numpy as np
import opt_einsum as oe
import cotengra as ctg
from tensornetwork import TensorNetwork

class MPS(TensorNetwork):
    """class of MPS

    physical bond: 0, 1, ..., n-1
    virtual bond: n, n+1, ..., 2n

    Attributes:
        n (int) : the number of tensors
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor, [physical, virtual_left, virtual_right]

    """

    def __init__(self, tensors):
        self.n = len(tensors)
        self.edges = []
        self.edge_dims = dict()
        self.tensors = tensors
        for i in range(self.n):
            self.edges.append([i, i+self.n, i+self.n+1])
            self.edge_dims[i] = self.tensors[i].shape[0]
            if i != 0 and self.edge_dims[i+self.n] != self.tensors[i].shape[1]:
                    raise ValueError("the dim of virtual bond do not correspond")
            if i == 0:
                self.edge_dims[self.n] = self.tensors[i].shape[1]
            self.edge_dims[i+self.n+1] = self.tensors[i].shape[2]