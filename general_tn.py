import numpy as np
from numpy.lib.arraysetops import ediff1d
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn

class TensorNetwork():
    """base class of Tensor Network

    Attributes:
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor

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
        self.nodes = []
        self.edges = []

        edge_list = dict()

        for i in range(self.n):
            self.nodes.append(tn.Node(tensors[i], axis_names=[str(10*i+j) for j in range(tensors[i].ndim)]))
            for j in range(len(edges[i])):
                if edges[i][j] in edge_list:
                    if edge_list[edges[i][j]][2] == True:
                        raise ValueError("HyperEdge is not allowed")
                    conn = self.nodes[edge_list[edges[i][j]][0]][edge_list[edges[i][j]][1]] ^ self.nodes[i][j]
                    edge_list[edges[i][j]][2] = True
                    self.edges.append(conn)
                else:
                    edge_list[edges[i][j]] = [i, j, False]

    def contract(self, output_edge_order=None):
        """contract the whole Tensor Network destructively

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
        Returns:
            np.array: tensor after contraction
        """

        if output_edge_order == None:
            return tn.contractors.auto(self.nodes, ignore_edge_order=True).tensor

        return tn.contractors.auto(self.nodes, output_edge_order=output_edge_order).tensor

    def replace_tensors(self, tensor_indexes, r_tensors):
        """replace tensors
        
        Args:
            tensor_indexes (list of int): specify indexes of tensors which are replaced
            r_tensors (list of np.array): alternate tensors
        """

        for i, tidx in enumerate(tensor_indexes):
            if r_tensors[i].ndim != self.nodes[tidx].get_rank():
                raise ValueError("the ndim of replaced tensors do not correspond")
            self.nodes[tidx].set_tensor(r_tensors[i])