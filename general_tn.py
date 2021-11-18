import numpy as np
from numpy.lib.arraysetops import ediff1d
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
import functools
import itertools
from collections import Counter
from itertools import chain

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
            self.nodes.append(tn.Node(tensors[i], name=f"node {i}"))
            node_edges = self.nodes[i].get_all_edges()
            for j in range(len(edges[i])):
                if edges[i][j] in edge_list:
                    if edge_list[edges[i][j]][2] == True:
                        raise ValueError("HyperEdge is not allowed")
                    conn = tn.connect(self.nodes[edge_list[edges[i][j]][0]][edge_list[edges[i][j]][1]], self.nodes[i][j], name=f"edge {edges[i][j]}")
                    edge_list[edges[i][j]][2] = True
                    self.edges.append(conn)
                else:
                    edge_list[edges[i][j]] = [i, j, False]
                    node_edges[j].set_name(f"edge {edges[i][j]}")

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


    def __visualize_path(self, path, input_sets, output_set, size_dict, node_list, output_edge_order=None):
        edge_alpha = dict()
        edge_alpha_dims = dict()
        alpha_offset = 0
        for node in node_list:
            for e in node.edges:
                if not e in edge_alpha:
                    edge_alpha[e] = oe.get_symbol(alpha_offset)
                    alpha_offset += 1
                    edge_alpha_dims[edge_alpha[e]] = size_dict[e]

        input_alpha = []
        for node in node_list:
            str = ""
            for e in node.edges:
                str += edge_alpha[e]
            input_alpha.append(str)

        output_alpha = None
        if output_edge_order is not None:
            str = ""
            for e in output_edge_order:
                str += edge_alpha[e]
            output_alpha = str
        else:
            str = ""
            for e in output_set:
                str += edge_alpha[e]
            output_alpha = str

        input_edge = [node.edges for node in node_list]
        total_cost = 0
        max_sp_cost = 0
        cost_list = []
        sp_cost_list = []
        einsum_str_list = []
        contract_einsum_str_list = []

        for cidx, contract_inds in enumerate(path):
            tensor_edge_list = [input_edge[c] for c in contract_inds]
            tensor_edge_alpha = [input_alpha[c] for c in contract_inds]
            freq = Counter(chain.from_iterable(tensor_edge_list))
            remain_edge_list = []
            contract_edge_list = []

            for idx in freq:
                if freq[idx] == 1:
                    remain_edge_list.append(idx)
                else:
                    contract_edge_list.append(idx)
            
            remain_edge_alpha = ""
            cost = 1
            sp_cost = 1
            for re in remain_edge_list:
                remain_edge_alpha += edge_alpha[re]
                cost *= size_dict[re]
                sp_cost *= size_dict[re]
            for ce in contract_edge_list:
                cost *= size_dict[ce]

            if cidx == len(path) - 1:
                remain_edge_list = output_edge_order
                remain_edge_alpha = output_alpha
            
            einsum_str = ",".join(input_alpha) + "->" + remain_edge_alpha
            contract_einsum_str = ",".join(tensor_edge_alpha) + "->" + remain_edge_alpha
            #print(f"{cidx}: {einsum_str}")
            #print(f"contract {contract_einsum_str} cost:{cost}")
            total_cost += cost
            max_sp_cost = max(max_sp_cost, sp_cost)
            cost_list.append(cost)
            sp_cost_list.append(sp_cost)
            einsum_str_list.append(einsum_str)
            contract_einsum_str_list.append(contract_einsum_str)

            #右側のindexから消去
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            for c in contract_inds:
                del input_alpha[c]
                del input_edge[c]
            input_alpha.append(remain_edge_alpha)
            input_edge.append(remain_edge_list)
        
        return einsum_str_list, contract_einsum_str_list, edge_alpha_dims, cost_list, total_cost, sp_cost_list, max_sp_cost


    def calc_contract_path(self, node_list, algorithm=None, memory_limit=2**28, output_edge_order=None, visualize=False):
        """calc contract path for given input and algorithm

        Args:
            node_list (list of tn.Node) : the nodes contracted
            algorithm : the path calculator
            memory_limit (int) : memory limit, default 2**28
            output_edge_order (list of tn.Edge) : the order of output edge
            visualize (bool) : if or not visualize contraction process
        
        Returns:
            oath (list of tuple of int) : the contraction path
            cost (int) : the total contraction cost
        """

        input_sets = [set(node.edges) for node in node_list]
        output_set = set()
        for edge in tn.get_all_edges(node_list):
            if edge.is_dangling() or not set(edge.get_nodes()) <= set(node_list):
                output_set.add(edge)
        size_dict = {edge: edge.dimension for edge in tn.get_all_edges(node_list)}

        if algorithm == None:
            algorithm = functools.partial(oe.paths.greedy, memory_limit=memory_limit)
        else:
            algorithm = functools.partial(oe.paths.optimal, memory_limit=None)

        path = algorithm(input_sets, output_set, size_dict)

        einsum_str_list, contract_einsum_str_list, edge_alpha_dims, cost_list, total_cost, sp_cost_list, max_sp_cost = self.__visualize_path(path, input_sets, output_set, size_dict, node_list, output_edge_order)
           
        if visualize: 
            print(f"path: {path}")
            print("edge_dims:", edge_alpha_dims)
            for i in range(len(path)):
                print(f"{i}: {einsum_str_list[i]}")
                print(f"contract {contract_einsum_str_list[i]} cost:{cost_list[i]} sp_cost:{sp_cost_list[i]}")
            print(f"estimated total cost: {total_cost} max sp cost: {max_sp_cost}")

        return path, total_cost


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

    @property
    def tensors(self):
        tensor_list = []
        for node in self.nodes:
            tensor_list.append(node.tensor)
        return tensor_list