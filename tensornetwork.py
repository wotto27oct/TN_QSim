import numpy as np
import opt_einsum as oe
import cotengra as ctg

class TensorNetwork():
    """base class of Tensor Network

    Attributes:
        n (int) : the number of tensors
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor

    """

    def __init__(self, edges, tensors):
        """initialization of Tensor Network
        Args:
            edges (list of list of int): the orderd indexes of each edge connected to each tensor
            tensors (list of np.array) : each tensor

        """
        self.n = len(tensors)
        if len(edges) != self.n:
            raise ValueError("the length of edges and tensors do not correspond")
        self.edges = edges
        self.edge_dims = dict()
        self.tensors = tensors
        self.is_dangling = dict()
        for i in range(self.n):
            for order in range(self.tensors[i].ndim):
                if self.edges[i][order] in self.edge_dims:
                    if self.edge_dims[self.edges[i][order]] != self.tensors[i].shape[order]:
                        raise ValueError("the dim of tensors do not correspond")
                    elif self.is_dangling[self.edges[i][order]]:
                        raise ValueError("Edge is duplicated")
                    self.is_dangling[self.edges[i][order]] = True
                else:
                    self.edge_dims[self.edges[i][order]] = self.tensors[i].shape[order]
                    self.is_dangling[self.edges[i][order]] = False

    def contract(self):
        """contract the whole Tensor Network

        all edges which dim is 1 is excluded.
        
        Returns:
            np.array: tensor after contraction
        """
        contract_args = []
        for i in range(self.n):
            new_shape = []
            new_edge = []
            for order in range(self.tensors[i].ndim):
                if self.edge_dims[self.edges[i][order]] != 1:
                    new_shape.append(self.edge_dims[self.edges[i][order]])
                    new_edge.append(self.edges[i][order])
            contract_args.append(self.tensors[i].reshape(new_shape))
            contract_args.append(new_edge)
        opt = ctg.HyperOptimizer()
        return oe.contract(*contract_args, optimize=opt)

    def replace_tensors(self, tensor_indexes, r_tensors):
        """replace tensors
        
        Args:
            tensor_indexes (list of int): specify indexes of tensors which are replaced
            r_tensors (list of np.array): alternate tensors
        """
        for i, tidx in enumerate(tensor_indexes):
            if r_tensors[i].ndim != len(self.edges[tidx]):
                raise ValueError("the ndim of replaced tensors do not correspond")
            self.tensors[tidx] = r_tensors[i]
            for idx, eidx in enumerate(self.edges[tidx]):
                self.edge_dims[eidx] = self.tensors[tidx].shape[idx]

        # check if edge_dims is not contradicted
        for tidx, tensor in enumerate(self.tensors):
            for order in range(tensor.ndim):
                if tensor.shape[order] != self.edge_dims[self.edges[tidx][order]]:
                    raise ValueError("the dims of replaced tensors do not correspond ")